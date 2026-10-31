module DifferentialGamesBenchmarks

using LinearAlgebra
import DifferentialGamesBase

# ============================================================================
# Include order — dependency constraints:
#
#  1. identity/case_id.jl            — CaseID
#  2. access/access_spec.jl          — AccessSpec hierarchy
#  3. identity/benchmark_case.jl     — BenchmarkCase (needs CaseID, AccessSpec)
#  4. challenges/*.jl                — ChallengeProfile + transform functions
#  5. evaluation/evaluator.jl        — Evaluator + independent metrics
#  6. orchestration/budget.jl        — Budget
#  7. orchestration/experiment.jl    — Experiment, Benchmark, run
#     (needs BenchmarkCase, ChallengeProfile, AccessSpec, Budget, Evaluator)
#  8. orchestration/results.jl       — ExperimentResult (needs Experiment)
#  9. registry/case_registry.jl      — CASE_REGISTRY, CASE_VISUALIZERS
# 10. viz_interface.jl               — visual_spec/visual_model extension points
# 11. cases/common.jl                — shared 2-player pursuit-evasion helpers
# 12. cases/*.jl                     — Classical Games pack
# ============================================================================

include("identity/case_id.jl")
include("access/access_spec.jl")
include("identity/benchmark_case.jl")
include("challenges/challenge_profile.jl")
include("challenges/stochastic_challenge.jl")
include("challenges/robust_challenge.jl")
include("challenges/po_challenge.jl")
include("challenges/communication_challenge.jl")
include("evaluation/evaluator.jl")
include("orchestration/budget.jl")
include("orchestration/experiment.jl")
include("orchestration/results.jl")
include("registry/case_registry.jl")
include("viz_interface.jl")
include("cases/common.jl")
include("cases/simple_pursuit.jl")
include("cases/homicidal_chauffeur.jl")
include("cases/two_cars.jl")
include("cases/lady_in_the_lake.jl")
include("cases/lq_formation.jl")
include("cases/hallway_passing.jl")
include("cases/racing.jl")
include("cases/falcon_suite.jl")

export
    CaseID,
    BenchmarkCase,
    game,
    ChallengeProfile,
    NOMINAL_CHALLENGE,
    apply,
    add_process_noise,
    add_disturbance_bound,
    restrict_observation,
    restrict_communication,
    AccessSpec,
    FullModelAccess,
    ApproximateModelAccess,
    SimulatorOnlyAccess,
    ModelFreeAccess,
    MaskedObjectives,
    ObservationOnlyAccess,
    access,
    Evaluator,
    default_evaluator,
    evaluate,
    validity_constraint_satisfaction,
    game_quality_nash_residual,
    robustness_metric_dynamics_residual,
    efficiency_metrics_from_solver_info,
    Budget,
    Experiment,
    Benchmark,
    run,
    ExperimentResult,
    register_case!,
    lookup_case,
    CASE_REGISTRY,
    register_visualizer!,
    visual_model_for,
    CASE_VISUALIZERS,
    visual_spec,
    visual_model,
    gameplot,
    dashboard,
    renderframe,
    animate,
    interactive,
    two_player_positions,
    two_player_separation,
    capture_radius_min,
    time_to_capture,
    escape_flag,
    capture_evaluator,
    two_agent_visual_model,
    SIMPLE_PURSUIT_NOMINAL_V1,
    SIMPLE_PURSUIT_CAPTURE_RADIUS,
    simple_pursuit_evaluator,
    HOMICIDAL_CHAUFFEUR_NOMINAL_V1,
    HOMICIDAL_CHAUFFEUR_CAPTURE_RADIUS,
    homicidal_chauffeur_evaluator,
    GAME_OF_TWO_CARS_NOMINAL_V1,
    GAME_OF_TWO_CARS_CAPTURE_RADIUS,
    game_of_two_cars_evaluator,
    LADY_IN_THE_LAKE_NOMINAL_V1,
    LADY_IN_THE_LAKE_RADIUS,
    lady_in_the_lake_evaluator,
    lady_in_the_lake_landing_metrics,
    LQ_FORMATION_NOMINAL_V1,
    LQ_FORMATION_OFFSET,
    lq_formation_evaluator,
    lq_formation_error,
    HALLWAY_PASSING_NOMINAL_V1,
    HallwayParameters,
    hallway_game,
    hallway_warmstart,
    hallway_centerline,
    hallway_halfwidth,
    hallway_starts_goals,
    hallway_goal_error,
    hallway_min_separation,
    hallway_wall_clearance,
    hallway_passing_evaluator,
    RACING_AUSTIN_NOMINAL_V1,
    RacingParameters,
    RaceTrack,
    austin_track,
    load_race_track,
    track_tangent,
    track_normal,
    track_errors,
    bicycle_dynamics,
    racing_starts,
    racing_collision_distance,
    racing_game,
    racing_warmstart,
    racing_progress,
    racing_min_separation,
    racing_track_clearance,
    racing_lag,
    racing_evaluator,
    austin_hairpin_track,
    austin_benchmark_initial_conditions,
    racing_game_from_ic,
    RACING_AUSTIN_HAIRPIN_V1,
    HALLWAY_PASSING_COMM_V1

end
