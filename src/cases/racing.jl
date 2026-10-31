# ============================================================================
# cases/racing.jl
#
# Two-car head-to-head racing on a segment of the Circuit of the Americas
# (Austin, tenth-scale), the benchmark of Zhu & Borrelli ("Dynamic Games
# Sequential Quadratic Programming", DGSQP) as ported in the FALCON validation
# suite (Outland & Arya, 2026).
#
#   state    q = [v_x, v_y, ψ̇, x, y, ψ, s]   dynamic bicycle (Pacejka tyres,
#                                              all-wheel drive) + progress s
#   control  u = [a, γ, u_s]                  acceleration, steering, progress rate
#
# The progress state s is decoupled from (x, y) by the dynamics (ṡ = u_s); a
# contouring/lag (MPCC) cost keeps it tied to the car's position on the track.
# Competition enters through the terminal cost: each car is paid for its own
# progress and for the margin by which it is ahead of the other,
#     J_i = ∫ ½ u'Wu + ½(q_c e_c² + q_l e_l²) dt  −  w_p s_i(T) + w_c (s_j(T) − s_i(T)).
# Collision avoidance (bounding circle), steering/acceleration bounds and the
# track boundaries are hard constraints.
#
# Port notes (differences from the FALCON script):
#   * The track is a natural cubic spline written in this file (no Interpolations
#     dependency), stored in data/f1_austin_segment.csv (s ∈ [60, 80] m of the
#     tenth-scale track, re-based to 0).
#   * Dynamics are the continuous bicycle ODE; the solver integrates them (the
#     FALCON script wraps an RK4 map with 10 substeps).
#   * Input-rate penalties and rate bounds are omitted: they need the previous
#     control, which only some solvers expose.
#   * Costs and constraints select a player's own block from the joint vector,
#     and the opponent's progress is read from the joint terminal state. A
#     solver that passes only the player's own block pays the progress cost only.
# ============================================================================

# ── Cubic spline (natural, C²), generic in the argument so it differentiates ─
struct CubicSpline1D
    t::Vector{Float64}
    y::Vector{Float64}
    M::Vector{Float64}   # second derivatives at the knots
end

function CubicSpline1D(t::AbstractVector, y::AbstractVector)
    n = length(t)
    M = zeros(n)
    if n > 2
        h = diff(t)
        A = zeros(n - 2, n - 2); rhs = zeros(n - 2)
        for i in 1:n-2
            A[i, i] = 2 * (h[i] + h[i+1])
            i > 1     && (A[i, i-1] = h[i])
            i < n - 2 && (A[i, i+1] = h[i+1])
            rhs[i] = 6 * ((y[i+2] - y[i+1]) / h[i+1] - (y[i+1] - y[i]) / h[i])
        end
        M[2:n-1] = A \ rhs
    end
    return CubicSpline1D(collect(Float64, t), collect(Float64, y), M)
end

function _interval(sp::CubicSpline1D, s)
    s = clamp(s, sp.t[1], sp.t[end])
    i = clamp(searchsortedlast(sp.t, s), 1, length(sp.t) - 1)
    return i, s
end

function (sp::CubicSpline1D)(s)
    i, s = _interval(sp, s)
    h = sp.t[i+1] - sp.t[i]
    a = (sp.t[i+1] - s) / h
    b = (s - sp.t[i]) / h
    return a * sp.y[i] + b * sp.y[i+1] + ((a^3 - a) * sp.M[i] + (b^3 - b) * sp.M[i+1]) * h^2 / 6
end

"""
    spline_derivative(sp, s)

First derivative of a [`CubicSpline1D`](@ref) at `s`.
"""
function spline_derivative(sp::CubicSpline1D, s)
    i, s = _interval(sp, s)
    h = sp.t[i+1] - sp.t[i]
    a = (sp.t[i+1] - s) / h
    b = (s - sp.t[i]) / h
    return (sp.y[i+1] - sp.y[i]) / h - (3a^2 - 1) / 6 * h * sp.M[i] + (3b^2 - 1) / 6 * h * sp.M[i+1]
end

"""
    RaceTrack

A track described by a C² centre-line `(x(s), y(s))`, arc-length parametrised by
`s ∈ [0, length]`, with left and right half-widths `left(s)`, `right(s)`.
"""
struct RaceTrack
    x::CubicSpline1D
    y::CubicSpline1D
    left::CubicSpline1D
    right::CubicSpline1D
    length::Float64
end

"""
    load_race_track(path) -> RaceTrack

Read a CSV with columns `s, x, y, left_width, right_width` (lines starting with `#` are comments).
"""
function load_race_track(path::AbstractString)
    rows = [parse.(Float64, split(l, ',')) for l in eachline(path)
            if !isempty(strip(l)) && !startswith(l, '#')]
    D = reduce(hcat, rows)'
    s = D[:, 1]
    return RaceTrack(CubicSpline1D(s, D[:, 2]), CubicSpline1D(s, D[:, 3]),
                     CubicSpline1D(s, D[:, 4]), CubicSpline1D(s, D[:, 5]), s[end])
end

"""
    austin_track() -> RaceTrack

The 20 m tenth-scale Circuit-of-the-Americas segment used by the benchmark.
"""
austin_track() = load_race_track(joinpath(@__DIR__, "..", "..", "data", "f1_austin_segment.csv"))

"""
    track_tangent(track, s) -> (tx, ty)

Unit tangent of the centre-line at arc length `s`.
"""
function track_tangent(tr::RaceTrack, s)
    dx, dy = spline_derivative(tr.x, s), spline_derivative(tr.y, s)
    d = sqrt(dx^2 + dy^2)
    return dx / d, dy / d
end

track_normal(tr::RaceTrack, s) = ((tx, ty) = track_tangent(tr, s); (-ty, tx))

"""
    track_errors(track, x, y, s) -> (e_c, e_l, e_y)

Contouring error `e_c`, lag error `e_l` and signed lateral offset `e_y` (positive to the left) of the
point `(x, y)` relative to the centre-line point at arc length `s`.
"""
function track_errors(tr::RaceTrack, x, y, s)
    tx, ty = track_tangent(tr, s)
    dx, dy = x - tr.x(s), y - tr.y(s)
    return ty * dx - tx * dy, -tx * dx - ty * dy, tx * dy - ty * dx
end

"""
    RacingParameters

Vehicle, cost and discretisation parameters of the racing benchmark (tenth-scale car).
"""
Base.@kwdef struct RacingParameters
    N::Int = 50
    dt::Float64 = 0.05
    l_f::Float64 = 0.18
    l_r::Float64 = 0.18
    mass::Float64 = 70.0
    I_z::Float64 = 2.0627
    μ::Float64 = 0.9
    pac_B::Float64 = 10.0
    pac_C::Float64 = 2.5
    u_max::Vector{Float64} = [2.1, 0.436, 4.0]
    u_min::Vector{Float64} = [-2.1, -0.436, 0.0]
    car_length::Float64 = 0.56
    car_width::Float64 = 0.2
    w_input::Vector{Float64} = [1.0, 1.0, 1e-4]
    w_prog::Float64 = 1.0
    w_comp::Float64 = 5.0
    q_c::Float64 = 0.1
    q_l::Float64 = 20.0
    v0::Float64 = 2.0
    ey_offset::Float64 = 0.3
end

"""
    racing_collision_distance(p::RacingParameters) -> Float64

Minimum centre distance between the two cars (twice the radius of the car's bounding circle).
"""
racing_collision_distance(p::RacingParameters) = 2 * sqrt((p.car_length / 2)^2 + (p.car_width / 2)^2)

"""
    bicycle_dynamics(p::RacingParameters) -> f

Continuous-time dynamic-bicycle right-hand side `f(q, u, _, _)` for `q = [v_x, v_y, ψ̇, x, y, ψ, s]`
and `u = [a, γ, u_s]`, with Pacejka lateral tyre forces.
"""
function bicycle_dynamics(p::RacingParameters)
    g = 9.81
    L = p.l_f + p.l_r
    Df = p.μ * p.mass * g * p.l_r / L
    Dr = p.μ * p.mass * g * p.l_f / L
    return (q, u, _, _) -> begin
        vx, vy, ψd, _, _, ψ, _ = q
        a, γ, us = u
        αf = -atan((vy + p.l_f * ψd) * cos(γ) - vx * sin(γ), vx * cos(γ) + (vy + p.l_f * ψd) * sin(γ))
        αr = -atan(vy - p.l_r * ψd, vx)
        Fyf = Df * sin(p.pac_C * atan(p.pac_B * αf))
        Fyr = Dr * sin(p.pac_C * atan(p.pac_B * αr))
        [a / 2 + a / 2 * cos(γ) - Fyf * sin(γ) / p.mass + ψd * vy,
         a / 2 * sin(γ) + (Fyf * cos(γ) + Fyr) / p.mass - ψd * vx,
         (p.l_f * Fyf * cos(γ) - p.l_r * Fyr) / p.I_z,
         vx * cos(ψ) - vy * sin(ψ),
         vy * cos(ψ) + vx * sin(ψ),
         ψd,
         us]
    end
end

"""
    racing_starts(track, p::RacingParameters) -> (q1, q2)

Side-by-side start: both cars at `s = 0`, lateral offsets `±ey_offset`, speed `v0`.
"""
function racing_starts(track::RaceTrack, p::RacingParameters)
    tx, ty = track_tangent(track, 0.0)
    nx, ny = track_normal(track, 0.0)
    ψ0 = atan(ty, tx)
    mk(side) = [p.v0, 0.0, 0.0, track.x(0.0) + side * p.ey_offset * nx,
                track.y(0.0) + side * p.ey_offset * ny, ψ0, 0.0]
    return mk(+1), mk(-1)
end

"""
    racing_game(track = austin_track(), p = RacingParameters()) -> GameProblem

Two-car racing PD-GNEP. See the file header for the model, cost and constraints. `q0 = (q¹, q²)` overrides the
side-by-side start with the two cars' initial states.
"""
function racing_game(track::RaceTrack = austin_track(), p::RacingParameters = RacingParameters(); q0 = nothing)
    n_x, n_u, n_p = 7, 3, 2
    f = bicycle_dynamics(p)
    q0 = q0 === nothing ? racing_starts(track, p) : (Vector{Float64}(q0[1]), Vector{Float64}(q0[2]))
    dmin2 = racing_collision_distance(p)^2

    players = map(1:n_p) do i
        ox, ou = n_x * (i - 1), n_u * (i - 1)
        j = 3 - i
        own_x(x) = length(x) == n_x * n_p ? x[ox+1:ox+n_x] : x
        own_u(u) = length(u) == n_u * n_p ? u[ou+1:ou+n_u] : u
        opp_s(x) = length(x) == n_x * n_p ? x[n_x*(j-1)+7] : own_x(x)[7]

        track_cost(q) = ((ec, el, _) = track_errors(track, q[4], q[5], q[7]);
                         0.5 * (p.q_c * ec^2 + p.q_l * el^2))
        stage = DifferentialGamesBase.NonlinearStageCost((x, u, _, _) -> begin
            q, ui = own_x(x), own_u(u)
            0.5 * sum(p.w_input .* ui .^ 2) + track_cost(q)
        end)
        term = DifferentialGamesBase.NonlinearTerminalCost((x, _) -> begin
            q = own_x(x)
            -p.w_prog * q[7] + p.w_comp * (opp_s(x) - q[7]) + track_cost(q)
        end)

        bounds = DifferentialGamesBase.ControlBounds(i; control_offset = 0, control_dim = n_u,
                                                      lower = p.u_min, upper = p.u_max)
        boundary = DifferentialGamesBase.PrivateInequality(i;
            func = (x, _, _, _) -> begin
                q = own_x(x)
                _, _, ey = track_errors(track, q[4], q[5], q[7])
                [ey - track.left(q[7]), -ey - track.right(q[7])]
            end, dim = 2)

        DifferentialGamesBase.PlayerSpec(i, n_x, n_u, q0[i], f,
            DifferentialGamesBase.PlayerObjective(i, stage, term), [bounds, boundary])
    end

    collision = DifferentialGamesBase.SharedInequality([1, 2];
        func = (x, _, _, _) -> [dmin2 - ((x[4] - x[n_x+4])^2 + (x[5] - x[n_x+5])^2)], dim = 1)

    return DifferentialGamesBase.PDGNEProblem(players,
        DifferentialGamesBase.AbstractSharedConstraint[collision],
        Float64(p.N * p.dt), Float64(p.dt))
end

"""
    racing_warmstart(track = austin_track(), p = RacingParameters(); v_ref = 2.5) -> WarmstartData

Closed-loop centre-line plan: each car is simulated with its own dynamics under a
simple controller (curvature feed-forward plus lateral and heading feedback to hold its initial
lateral offset, proportional speed control to `v_ref`) and the controls that were applied are
returned; the progress rate is set to the car's forward speed so that `s` stays tied to its
position. Only the controls are used by `ALGAMES`.
"""
function racing_warmstart(track::RaceTrack = austin_track(), p::RacingParameters = RacingParameters();
                          v_ref::Real = 2.5, k_ey::Real = 0.5, k_ψ::Real = 1.0)
    f = bicycle_dynamics(p)
    L = p.l_f + p.l_r
    κ(s) = begin
        h = 1e-3; s1, s2 = max(s - h, 0.0), min(s + h, track.length)
        t1, t2 = track_tangent(track, s1), track_tangent(track, s2)
        (t1[1] * t2[2] - t1[2] * t2[1]) / (s2 - s1)
    end
    rk4(q, u, h) = (k1 = f(q, u, nothing, 0); k2 = f(q + h / 2 * k1, u, nothing, 0);
                    k3 = f(q + h / 2 * k2, u, nothing, 0); k4 = f(q + h * k3, u, nothing, 0);
                    q + h / 6 * (k1 + 2k2 + 2k3 + k4))
    trajs = DifferentialGamesBase.Trajectory{Float64}[]
    for (i, q) in enumerate(racing_starts(track, p))
        _, _, ey_ref = track_errors(track, q[4], q[5], q[7])
        U = zeros(3, p.N)
        for k in 1:p.N
            s = clamp(q[7], 0.0, track.length)
            tx, ty = track_tangent(track, s)
            _, _, ey = track_errors(track, q[4], q[5], s)
            eψ = mod(q[6] - atan(ty, tx) + π, 2π) - π
            γ = clamp(atan(κ(s) * L) - k_ey * (ey - ey_ref) - k_ψ * eψ, -p.u_max[2], p.u_max[2])
            a = clamp(2.0 * (v_ref - q[1]), p.u_min[1], p.u_max[1])
            u = [a, γ, clamp(q[1], p.u_min[3], p.u_max[3])]
            U[:, k] = u
            for _ in 1:5
                q = rk4(q, u, p.dt / 5)
            end
        end
        push!(trajs, DifferentialGamesBase.Trajectory{Float64}(i, zeros(7, p.N + 1), U,
                                                               collect(0.0:p.dt:p.N*p.dt), 0.0))
    end
    return DifferentialGamesBase.WarmstartData{Float64}(trajs, nothing, Dict{Symbol,Any}())
end

# ── Independent metrics ─────────────────────────────────────────────────────
function _racing_states(game, sol)
    fg = _forward_game(game)
    Xm, _, _ = _joint_state_control_matrices(game, sol)
    return [Xm[fg.metadata.state_offsets[i]+1:fg.metadata.state_offsets[i]+7, :] for i in 1:2]
end

"""
    racing_progress(game, sol) -> (s_1, s_2)

Final progress of each car along the centre-line, in metres.
"""
racing_progress(game, sol::DifferentialGamesBase.GNEPSolution) =
    Tuple(Q[7, end] for Q in _racing_states(game, sol))

"""
    racing_min_separation(game, sol) -> Float64

Smallest centre distance between the cars over the horizon.
"""
function racing_min_separation(game, sol::DifferentialGamesBase.GNEPSolution)
    Q1, Q2 = _racing_states(game, sol)
    return minimum(hypot(Q1[4, k] - Q2[4, k], Q1[5, k] - Q2[5, k]) for k in axes(Q1, 2))
end

"""
    racing_track_clearance(game, sol, track) -> Float64

Smallest distance of either car's centre to a track edge (negative means it left the track).
"""
function racing_track_clearance(game, sol::DifferentialGamesBase.GNEPSolution, track::RaceTrack)
    return minimum(minimum(begin
        _, _, ey = track_errors(track, Q[4, k], Q[5, k], Q[7, k])
        min(track.left(Q[7, k]) - ey, track.right(Q[7, k]) + ey)
    end for k in axes(Q, 2)) for Q in _racing_states(game, sol))
end

"""
    racing_lag(game, sol, track) -> Float64

Largest distance between a car and the centre-line point at its own progress `s` over the horizon
(how well the progress state stays tied to the car's position).
"""
racing_lag(game, sol::DifferentialGamesBase.GNEPSolution, track::RaceTrack) =
    maximum(maximum(hypot(Q[4, k] - track.x(Q[7, k]), Q[5, k] - track.y(Q[7, k])) for k in axes(Q, 2))
            for Q in _racing_states(game, sol))

const RACING_AUSTIN_NOMINAL_V1 = BenchmarkCase{Float64}(
    CaseID(:Racing, :AustinNominal, v"1"),
    () -> racing_game(),
    FullModelAccess(),
    "Two-car head-to-head racing on a 20 m segment of the Austin tenth-scale track: " *
    "7-state dynamic bicycle with Pacejka tyres, competitive terminal progress, " *
    "contouring/lag cost, bounding-circle collision avoidance and track-boundary constraints.",
    ["Zhu, E. L. & Borrelli, F. (2024). A sequential quadratic programming approach to the solution of open-loop generalized Nash equilibria for autonomous racing.",
     "Outland, B. & Arya, H. A Fast, Convergent Algorithm for Solving Non-convex Partially-Decoupled Generalized Nash Equilibrium Problems (FALCON; racing benchmark as in the DGSQP comparison)."],
)
register_case!(RACING_AUSTIN_NOMINAL_V1)

"""
    racing_evaluator(; track = austin_track()) -> Evaluator

`default_evaluator()` extended with `:progress` (final `s` of each car), `:min_separation`,
`:track_clearance` and `:lag` (distance of each car from the centre-line point at its own `s`).
"""
function racing_evaluator(; track::RaceTrack = austin_track())
    ev = default_evaluator()
    ev.quality_metrics[:progress]        = (g, s) -> racing_progress(g, s)
    ev.quality_metrics[:min_separation]  = (g, s) -> racing_min_separation(g, s)
    ev.quality_metrics[:track_clearance] = (g, s) -> racing_track_clearance(g, s, track)
    ev.quality_metrics[:lag]             = (g, s) -> racing_lag(g, s, track)
    return ev
end

"""
    racing_visual_model(game, sol; track = austin_track()) -> NamedTuple

Car paths and the two track edges (`walls`), for `gameplot`.
"""
function racing_visual_model(game, sol::DifferentialGamesBase.GNEPSolution; track::RaceTrack = austin_track())
    Qs = _racing_states(game, sol)
    _, _, times = _joint_state_control_matrices(game, sol)
    agents = [(player_id = i, label = "Car $i", positions = Qs[i][4:5, :]) for i in 1:2]
    ss = collect(range(0.0, track.length; length = 300))
    edge(sgn) = (x = [track.x(s) + sgn * (sgn > 0 ? track.left(s) : track.right(s)) * track_normal(track, s)[1] for s in ss],
                 y = [track.y(s) + sgn * (sgn > 0 ? track.left(s) : track.right(s)) * track_normal(track, s)[2] for s in ss])
    return (agents = agents, walls = [edge(+1), edge(-1)], times = times)
end

register_visualizer!(RACING_AUSTIN_NOMINAL_V1.id, racing_visual_model)
