# ============================================================================
# cases/hallway_passing.jl
#
# Multi-agent S-shaped double-choke hallway passing, ported from the FALCON
# validation suite (Outland & Arya, 2026). Three unicycle agents start
# dispersed on the left of a corridor with two narrow chokes whose centre-lines
# are offset in opposite directions, and must reach goals on the right while
# staying clear of the walls and of each other.
#
# Geometry (x in [-1, 1], smooth so that gradients exist everywhere):
#     y_c(x) = -y_bump * sin(pi x)                        centre-line
#     w(x)   = w_narrow + (w_wide - w_narrow) cos^2(pi x) half-width
# The choke half-width satisfies 2 w_narrow < d_safe, so the agents are forced
# through each choke in single file; the S-shape forces them to swing up and then
# down. This is the "negotiation" that makes the problem non-trivial.
#
# Port notes (what differs from the FALCON script):
#   * Dynamics are the continuous-time unicycle (the FALCON script hands the
#     solver an Euler map); every solver in the ecosystem integrates the
#     continuous right-hand side itself.
#   * The hard terminal ball ||x_N - g|| <= eps_goal is replaced by the
#     paper's quadratic terminal cost rho_goal ||x_N - g||^2, which every
#     solver accepts; the evaluator reports the final goal error so reaching the
#     goal is still checked independently.
#   * The paper's soft proximity-repulsion cost (rho_p) is not included: the
#     hard collision constraint carries the separation.
#   * The communication-range constraint of the FALCON script is available as
#     the `d_comm` option (default off) because it only matters for the
#     communication-limited experiments.
# ============================================================================

"""
    HallwayParameters

Geometry and weights of the S-shaped double-choke hallway. See
[`hallway_game`](@ref) for the meaning of the fields.
"""
Base.@kwdef struct HallwayParameters
    N::Int            = 100
    dt::Float64       = 0.1
    y_bump::Float64   = 0.3
    w_wide::Float64   = 0.55
    w_narrow::Float64 = 0.08
    d_safe::Float64   = 0.17
    d_wall::Float64   = 0.06
    dy::Float64       = 0.2
    v_max::Float64    = 1.0
    ω_max::Float64    = π / 3
    ρ_goal::Float64   = 1000.0
    ρ_ctrl::Float64   = 1.0
    d_comm::Float64   = Inf
end

"""
    hallway_centerline(p::HallwayParameters, x) -> y_c(x)

Centre-line height of the corridor at longitudinal position `x`.
"""
hallway_centerline(p::HallwayParameters, x) = -p.y_bump * sin(π * x)

"""
    hallway_halfwidth(p::HallwayParameters, x) -> w(x)

Half-width of the corridor at longitudinal position `x` (before the wall inset `d_wall`).
"""
hallway_halfwidth(p::HallwayParameters, x) =
    p.w_narrow + (p.w_wide - p.w_narrow) * cos(π * x)^2

"""
    hallway_starts_goals(p::HallwayParameters) -> (starts, goals)

Initial states `[x, y, θ]` and goal states of the three agents.
"""
function hallway_starts_goals(p::HallwayParameters)
    starts = [[-0.9, -p.dy, 0.0], [-0.9, 0.0, 0.0], [-0.9, p.dy, 0.0]]
    goals  = [[ 0.9, -p.dy, 0.0], [ 0.9, 0.0, 0.0], [ 0.9, p.dy, 0.0]]
    return starts, goals
end

"""
    hallway_game(p::HallwayParameters = HallwayParameters()) -> GameProblem

Build the three-agent hallway game as a PD-GNEP: each agent is a unicycle
`ẋ = [v cos θ, v sin θ, ω]` with controls `(v, ω) ∈ [0, v_max] × [−ω_max, ω_max]`,
a private wall constraint, a shared pairwise collision constraint
`‖pᵢ − pⱼ‖ ≥ d_safe` and the cost
`ρ_ctrl ∫‖uᵢ‖² dt + ρ_goal ‖xᵢ(T) − gᵢ‖²` (the goal penalty is on the full state `[x, y, θ]`, as in the paper).

The cost and constraint functions select the player's own block when handed the
joint vector, so they give the same value whether a solver passes the joint or the
player's own state and control.
"""
function hallway_game(p::HallwayParameters = HallwayParameters())
    N, dt = p.N, p.dt
    tf = N * dt
    starts, goals = hallway_starts_goals(p)
    n_x, n_u, n_p = 3, 2, 3

    @assert p.w_narrow < p.w_wide
    @assert p.dy < p.w_wide
    @assert 2 * p.w_narrow < p.d_safe "single-file condition: 2*w_narrow < d_safe"
    @assert p.d_wall < p.w_narrow

    dyn = (x, u, _, _) -> [u[1] * cos(x[3]), u[1] * sin(x[3]), u[2]]

    players = map(1:n_p) do i
        ox, ou = n_x * (i - 1), n_u * (i - 1)
        own_x(x) = length(x) == n_x * n_p ? x[ox+1:ox+n_x] : x
        own_u(u) = length(u) == n_u * n_p ? u[ou+1:ou+n_u] : u
        gi = goals[i]

        stage = DifferentialGamesBase.NonlinearStageCost(
            (x, u, _, _) -> p.ρ_ctrl * sum(abs2, own_u(u)); is_separable = true)
        term = DifferentialGamesBase.NonlinearTerminalCost(
            (x, _) -> p.ρ_goal * sum(abs2, own_x(x) .- gi))

        wall = DifferentialGamesBase.PrivateInequality(i;
            func = (x, _, _, _) -> begin
                xo = own_x(x)
                yc = hallway_centerline(p, xo[1])
                w  = hallway_halfwidth(p, xo[1]) - p.d_wall
                [xo[2] - (yc + w), (yc - w) - xo[2]]
            end,
            dim = 2)

        bounds = DifferentialGamesBase.ControlBounds(i;
            control_offset = 0, control_dim = n_u,
            lower = [0.0, -p.ω_max], upper = [p.v_max, p.ω_max])

        DifferentialGamesBase.PlayerSpec(i, n_x, n_u, starts[i], dyn,
            DifferentialGamesBase.PlayerObjective(i, stage, term),
            [bounds, wall])
    end

    pair_dist2(x) = [sum(abs2, x[1:2] .- x[4:5]),
                     sum(abs2, x[1:2] .- x[7:8]),
                     sum(abs2, x[4:5] .- x[7:8])]
    shared = Any[DifferentialGamesBase.SharedInequality([1, 2, 3];
        func = (x, _, _, _) -> p.d_safe^2 .- pair_dist2(x), dim = 3)]
    isfinite(p.d_comm) && push!(shared, DifferentialGamesBase.SharedInequality([1, 2, 3];
        func = (x, _, _, _) -> pair_dist2(x) .- p.d_comm^2, dim = 3))

    return DifferentialGamesBase.PDGNEProblem(players, convert(Vector{DifferentialGamesBase.AbstractSharedConstraint}, shared),
                                              Float64(tf), Float64(dt))
end

"""
    hallway_warmstart(p::HallwayParameters = HallwayParameters(); order = (2, 3, 1), gap = 1.5) -> WarmstartData

A good (not exactly feasible) open-loop plan: every agent follows the corridor
centre-line through both chokes (blending from its start offset and to its goal
offset in the open sections), with
agent `order[k]` starting `(k-1)·gap` seconds after the first so that the chokes
are passed in single file. Only the controls are used by `ALGAMES`; they are
rolled out from the initial state.

The plan keeps agents on the centre-line through both chokes and reaches the
goals, but agents whose paths cross come within about 0.07 of each other (the
safe distance is 0.17) and the walls are touched by about 1 mm at the start,
so the solver still has to repair those. The problem is non-convex and the
straight-line initialisation crosses the walls, so a plan of this kind is how
the solvers are meant to be started on this case.
"""
function hallway_warmstart(p::HallwayParameters = HallwayParameters(); order = (2, 3, 1), gap = 1.5)
    N, dt = p.N, p.dt
    starts, goals = hallway_starts_goals(p)
    yc(x) = hallway_centerline(p, x)
    trajs = DifferentialGamesBase.Trajectory{Float64}[]
    for i in 1:3
        # Reference path y_ref(x): hold the start height, blend onto the corridor
        # centre-line before the first choke and off it onto the goal height after
        # the second (smoothstep blends, so the heading is 0 at the start and the
        # goal, as required by the boundary conditions).
        smooth(a) = (a = clamp(a, 0.0, 1.0); a^2 * (3 - 2a))
        a_in(x)  = smooth((x - starts[i][1]) / 0.35)
        a_out(x) = smooth((x - 0.55) / 0.35)
        ref_y(x) = (1 - a_in(x)) * starts[i][2] +
                   a_in(x) * ((1 - a_out(x)) * yc(x) + a_out(x) * goals[i][2])
        xs  = collect(range(starts[i][1], goals[i][1]; length = 2000))
        wps = [[x, ref_y(x)] for x in xs]
        seglen = [norm(wps[k+1] - wps[k]) for k in 1:length(wps)-1]
        cum = [0.0; cumsum(seglen)]
        L = cum[end]
        delay = (findfirst(==(i), order) - 1) * gap
        # Speed profile: nominal speed, slowed where the path curves so that the
        # turn rate stays below 85 % of its bound. Times are integrated along the path.
        θseg = [atan(wps[k+1][2] - wps[k][2], wps[k+1][1] - wps[k][1]) for k in 1:length(seglen)]
        κseg = [k == 1 ? 0.0 : abs(θseg[k] - θseg[k-1]) / seglen[k] for k in 1:length(seglen)]
        vprof = [min(0.6 * p.v_max, 0.85 * p.ω_max / (κ + 1e-9)) for κ in κseg]
        tcum = [0.0; cumsum(seglen ./ vprof)]
        σ(t) = begin
            τ = clamp(t - delay, 0.0, tcum[end])
            k = clamp(searchsortedlast(tcum, τ), 1, length(seglen))
            cum[k] + (τ - tcum[k]) * vprof[k]
        end
        heading(σ_) = begin
            k = clamp(searchsortedlast(cum, σ_), 1, length(seglen))
            atan(wps[k+1][2] - wps[k][2], wps[k+1][1] - wps[k][1])
        end
        # The unicycle is differentially flat: speed and turn rate follow from
        # the arc length and heading of the reference at consecutive nodes.
        U = zeros(2, N)
        for k in 1:N
            σ0, σ1 = σ((k - 1) * dt), σ(k * dt)
            U[1, k] = clamp((σ1 - σ0) / dt, 0.0, p.v_max)
            U[2, k] = clamp((heading(σ1) - heading(σ0)) / dt, -p.ω_max, p.ω_max)
        end
        push!(trajs, DifferentialGamesBase.Trajectory{Float64}(i, zeros(3, N + 1), U, collect(0.0:dt:N*dt), 0.0))
    end
    return DifferentialGamesBase.WarmstartData{Float64}(trajs, nothing, Dict{Symbol,Any}())
end

"""
    hallway_goal_error(game, sol) -> Float64

Largest final distance of any agent from its goal. Recomputed from the
trajectories, independent of the solver's convergence flag.
"""
function hallway_goal_error(game, sol::DifferentialGamesBase.GNEPSolution; params = HallwayParameters())
    _, goals = hallway_starts_goals(params)
    fg = _forward_game(game)
    Xm, _, _ = _joint_state_control_matrices(game, sol)
    return maximum(1:3) do i
        o = fg.metadata.state_offsets[i]
        norm(Xm[o+1:o+2, end] .- goals[i][1:2])
    end
end

"""
    hallway_min_separation(game, sol) -> Float64

Smallest pairwise distance between any two agents over the horizon.
"""
function hallway_min_separation(game, sol::DifferentialGamesBase.GNEPSolution)
    fg = _forward_game(game)
    Xm, _, _ = _joint_state_control_matrices(game, sol)
    offs = fg.metadata.state_offsets
    return minimum(minimum(norm(Xm[offs[i]+1:offs[i]+2, k] .- Xm[offs[j]+1:offs[j]+2, k])
                           for k in axes(Xm, 2))
                   for i in 1:3 for j in i+1:3)
end

"""
    hallway_wall_clearance(game, sol; params) -> Float64

Smallest signed clearance to either wall (after the `d_wall` inset); negative values mean the agent left the corridor.
"""
function hallway_wall_clearance(game, sol::DifferentialGamesBase.GNEPSolution; params = HallwayParameters())
    fg = _forward_game(game)
    Xm, _, _ = _joint_state_control_matrices(game, sol)
    offs = fg.metadata.state_offsets
    return minimum(1:3) do i
        minimum(axes(Xm, 2)) do k
            x, y = Xm[offs[i]+1, k], Xm[offs[i]+2, k]
            yc = hallway_centerline(params, x)
            w  = hallway_halfwidth(params, x) - params.d_wall
            min(yc + w - y, y - (yc - w))
        end
    end
end

const HALLWAY_PASSING_NOMINAL_V1 = BenchmarkCase{Float64}(
    CaseID(:HallwayPassing, :Nominal, v"1"),
    () -> hallway_game(),
    FullModelAccess(),
    "Three unicycle agents negotiate an S-shaped double-choke hallway whose " *
    "chokes are narrower than the safe distance, forcing single-file passage " *
    "in opposite directions. Ported from the FALCON validation suite.",
    ["Outland, B. & Arya, H. A Fast, Convergent Algorithm for Solving Non-convex Partially-Decoupled Generalized Nash Equilibrium Problems (narrowing hallways problem)."],
)
register_case!(HALLWAY_PASSING_NOMINAL_V1)

"""
    hallway_passing_evaluator(; params = HallwayParameters()) -> Evaluator

`default_evaluator()` extended with hallway-specific metrics: `:goal_error`
(largest final distance to a goal), `:min_separation` (smallest pairwise
distance) and `:wall_clearance` (smallest signed distance to a wall).
"""
function hallway_passing_evaluator(; params = HallwayParameters())
    ev = default_evaluator()
    ev.quality_metrics[:goal_error]     = (g, s) -> hallway_goal_error(g, s; params)
    ev.quality_metrics[:min_separation] = (g, s) -> hallway_min_separation(g, s)
    ev.quality_metrics[:wall_clearance] = (g, s) -> hallway_wall_clearance(g, s; params)
    return ev
end

"""
    hallway_visual_model(game, sol; params = HallwayParameters()) -> NamedTuple

Agent paths and the two corridor walls (`walls`), for `gameplot`.
"""
function hallway_visual_model(game, sol::DifferentialGamesBase.GNEPSolution; params = HallwayParameters())
    fg = _forward_game(game)
    Xm, _, times = _joint_state_control_matrices(game, sol)
    offs = fg.metadata.state_offsets
    agents = [(player_id = i, label = "Agent $i", positions = Xm[offs[i]+1:offs[i]+2, :]) for i in 1:3]
    xs = collect(range(-1.0, 1.0; length = 400))
    walls = [(x = xs, y = [hallway_centerline(params, x) + sgn * hallway_halfwidth(params, x) for x in xs])
             for sgn in (+1, -1)]
    return (agents = agents, walls = walls, times = times)
end

register_visualizer!(HALLWAY_PASSING_NOMINAL_V1.id, hallway_visual_model)
