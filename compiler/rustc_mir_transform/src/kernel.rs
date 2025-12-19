use crate::{coverage, lint_tail_expr_drop_order, promote_consts, sanity_check, simplify};
use crate::{inline, run_analysis_to_runtime_passes};
use crate::abort_unwinding_calls::AbortUnwindingCalls;
use hir::ConstContext;
use crate::required_consts::RequiredConstsVisitor;
use rustc_const_eval::check_consts::{self, ConstCx};
use rustc_const_eval::util;
use rustc_data_structures::fx::FxIndexSet;
use rustc_data_structures::steal::Steal;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_index::IndexVec;
use rustc_middle::mir::{
    AnalysisPhase, Body, CallSource, ClearCrossCrate, ConstOperand, ConstQualifs, LocalDecl,
    MirPhase, Operand, Place, ProjectionElem, Promoted, RuntimePhase, Rvalue, START_BLOCK,
    SourceInfo, Statement, StatementKind, TerminatorKind,
};
use rustc_middle::ty::{self, TyCtxt, TypeVisitableExt};
use rustc_middle::util::Providers;
use rustc_middle::{bug, query, span_bug};
use rustc_mir_build::builder::build_mir;
use rustc_span::source_map::Spanned;
use rustc_span::{DUMMY_SP, sym};
use tracing::debug;

use std::sync::LazyLock;

use crate::pass_manager::{self as pm, Lint, MirLint, MirPass, WithMinOptLevel};

use crate::{kernel_lang_item_swap::KernelLangItemSwap};

// NOTE(jorik): hier alle kernel functies
pub fn optimize_generated_kernel_mir<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    RequiredConstsVisitor::compute_required_consts(body);
}

// NOTE(jorik): Lang item swap in kernel MIR
/// Optimize the MIR and prepare it for codegen.
/// specifically for kernel code
fn optimized_kernel_mir<'tcx>(tcx: TyCtxt<'tcx>, did: DefId) -> &'tcx Body<'tcx> {
    // get the normal optimized mir
    let mut body = tcx.optimized_mir(did).clone();

    let kernel_swap_pass = KernelLangItemSwap::new(tcx);
    kernel_swap_pass.run_pass(tcx, &mut body);

    // RemoveDropGlue.run_pass(tcx, &mut body);
    AbortUnwindingCalls.run_pass_for_device_code(tcx, &mut body);

    tcx.arena.alloc(body)
}

/// Obtain just the main MIR (no promoteds) and run some cleanups on it. This also runs
/// mir borrowck *before* doing so in order to ensure that borrowck can be run and doesn't
/// end up missing the source MIR due to stealing happening.
fn mir_drops_elaborated_and_const_checked_kernel(tcx: TyCtxt<'_>, def: DefId) -> &Steal<Body<'_>> {
    if tcx.is_coroutine(def) {
        //tcx.ensure_done().mir_coroutine_witnesses(def);
        panic!("Tried to compile coroutine in kernel");
    }

    // We only need to borrowck non-synthetic MIR.
    let tainted_by_errors = if !tcx.is_synthetic_mir(def) {
        tcx.mir_borrowck(tcx.typeck_root_def_id(def).expect_local()).err()
    } else {
        None
    };

    let is_fn_like = tcx.def_kind(def).is_fn_like();
    if is_fn_like {
        // Do not compute the mir call graph without said call graph actually being used.
        if pm::should_run_pass(tcx, &inline::Inline, pm::Optimizations::Allowed)
            || inline::ForceInline::should_run_pass_for_callee(tcx, def)
        {
            tcx.ensure_done().mir_inliner_callees(ty::InstanceKind::Item(def));
        }
    }

    let (body, _) = tcx.mir_promoted_kernel(def);
    let mut body = body.steal();

    if let Some(error_reported) = tainted_by_errors {
        body.tainted_by_errors = Some(error_reported);
    }

    // Also taint the body if it's within a top-level item that is not well formed.
    //
    // We do this check here and not during `mir_promoted` because that may result
    // in borrowck cycles if WF requires looking into an opaque hidden type.
    let root = tcx.typeck_root_def_id(def);
    match tcx.def_kind(root) {
        DefKind::Fn
        | DefKind::AssocFn
        | DefKind::Static { .. }
        | DefKind::Const
        | DefKind::AssocConst => {
            if let Err(guar) = tcx.ensure_ok().check_well_formed(root.expect_local()) {
                body.tainted_by_errors = Some(guar);
            }
        }
        _ => {}
    }

    run_analysis_to_runtime_passes(tcx, &mut body);

    tcx.alloc_steal_mir(body)
}

// NOTE(jorik): this is where constant promotion begins
/// Compute the main MIR body and the list of MIR bodies of the promoteds.
fn mir_promoted_kernel(
    tcx: TyCtxt<'_>,
    def: DefId,
) -> (&Steal<Body<'_>>, &Steal<IndexVec<Promoted, Body<'_>>>) {
    // Ensure that we compute the `mir_const_qualif` for constants at
    // this point, before we steal the mir-const result.
    // Also this means promotion can rely on all const checks having been done.

    let const_qualifs = match tcx.def_kind(def) {
        DefKind::Fn | DefKind::AssocFn | DefKind::Closure
            if tcx.constness(def) == hir::Constness::Const
                || tcx.is_const_default_method(def) =>
        {
            tcx.mir_const_qualif(def)
        }
        DefKind::AssocConst
        | DefKind::Const
        | DefKind::Static { .. }
        | DefKind::InlineConst
        | DefKind::AnonConst => tcx.mir_const_qualif(def),
        _ => ConstQualifs::default(),
    };

    // the `has_ffi_unwind_calls` query uses the raw mir, so make sure it is run.
    // tcx.ensure_done().has_ffi_unwind_calls(def);

    // the `by_move_body` query uses the raw mir, so make sure it is run.
    if tcx.needs_coroutine_by_move_body_def_id(def) {
        tcx.ensure_done().coroutine_by_move_body_def_id(def);
    }

    let mut body = tcx.mir_built_kernel(def).steal();
    if let Some(error_reported) = const_qualifs.tainted_by_errors {
        body.tainted_by_errors = Some(error_reported);
    }

    // Collect `required_consts` *before* promotion, so if there are any consts being promoted
    // we still add them to the list in the outer MIR body.
    // TODO GPU: add kernels to required consts?
    RequiredConstsVisitor::compute_required_consts(&mut body);

    // What we need to run borrowck etc.
    let promote_pass = promote_consts::PromoteTemps::default();
    pm::run_passes(
        tcx,
        &mut body,
        &[&promote_pass, &simplify::SimplifyCfg::PromoteConsts, &coverage::InstrumentCoverage],
        Some(MirPhase::Analysis(AnalysisPhase::Initial)),
        pm::Optimizations::Allowed,
    );

    //lint_tail_expr_drop_order::run_lint(tcx, def, &body);

    let promoted = promote_pass.promoted_fragments.into_inner();
    (tcx.alloc_steal_mir(body), tcx.alloc_steal_promoted(promoted))
}

fn mir_const_qualif_kernel(tcx: TyCtxt<'_>, def: DefId) -> ConstQualifs {
    // N.B., this `borrow()` is guaranteed to be valid (i.e., the value
    // cannot yet be stolen), because `mir_promoted()`, which steals
    // from `mir_built()`, forces this query to execute before
    // performing the steal.
    let body = &tcx.mir_built_kernel(def).borrow();
    let ccx = check_consts::ConstCx::new(tcx, body);
    // No need to const-check a non-const `fn`.
    match ccx.const_kind {
        Some(ConstContext::Const { .. } | ConstContext::Static(_) | ConstContext::ConstFn) => {}
        None => span_bug!(
            tcx.def_span(def),
            "`mir_const_qualif` should only be called on const fns and const items"
        ),
    }

    if body.return_ty().references_error() {
        // It's possible to reach here without an error being emitted (#121103).
        tcx.dcx().span_delayed_bug(body.span, "mir_const_qualif: MIR had errors");
        return Default::default();
    }

    let mut validator = check_consts::check::Checker::new(&ccx);
    validator.check_body();

    // We return the qualifs in the return place for every MIR body, even though it is only used
    // when deciding to promote a reference to a `const` for now.
    validator.qualifs_in_return_place()
}

// NOTE(jorik): this is where MIR is built
fn mir_built_kernel(tcx: TyCtxt<'_>, def: DefId) -> &Steal<Body<'_>> {
    let mut body = tcx.mir_base(def).clone();

    pm::run_passes(
        tcx,
        &mut body,
        &[
            // TODO(jorik): Add extra pass for target selection
            // What we need to do constant evaluation.
            &simplify::SimplifyCfg::Initial, // NOTE(jorik) separate into another stage
            &Lint(sanity_check::SanityCheck), // this one too
        ],
        None,
        pm::Optimizations::Allowed,
    );
    tcx.alloc_steal_mir(body)
}


pub fn provide(providers: &mut Providers) {
    providers.queries = query::Providers {
        mir_drops_elaborated_and_const_checked_kernel,
        optimized_kernel_mir,
        mir_promoted_kernel,
        mir_const_qualif_kernel,
        mir_built_kernel,
        ..providers.queries
    };
}
