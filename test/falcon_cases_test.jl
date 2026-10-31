using Test
using DifferentialGamesBenchmarks
using LinearAlgebra
import DifferentialGamesBase

# These cases are ported from the FALCON validation suite. They are hard,
# non-convex benchmarks that the reference solvers in this ecosystem do not
# solve to convergence, so the tests check the problem definition (geometry,
# dynamics, feasibility of the reference plan) rather than a solver outcome.

@testset "Hallway passing" begin
    P = HallwayParameters()

    @testset "geometry" begin
        # Chokes at x = ±0.5: width w_narrow, centre ±y_bump; open at x = 0, ±1
        @test hallway_halfwidth(P, 0.5) ≈ P.w_narrow
        @test hallway_halfwidth(P, -0.5) ≈ P.w_narrow
        @test hallway_halfwidth(P, 0.0) ≈ P.w_wide
        @test hallway_centerline(P, -0.5) ≈ P.y_bump
        @test hallway_centerline(P, 0.5) ≈ -P.y_bump
        # Single-file condition: two agents cannot pass a choke abreast
        @test 2 * P.w_narrow < P.d_safe
        starts, goals = hallway_starts_goals(P)
        for q in vcat(starts, goals)   # all start/goal positions are inside the walls
            @test hallway_halfwidth(P, q[1]) - P.d_wall - abs(q[2] - hallway_centerline(P, q[1])) > -5e-3   # within 5 mm of the inset wall
        end
        @test_throws AssertionError hallway_game(HallwayParameters(w_narrow = 0.2))
    end

    @testset "game" begin
        g = hallway_game(P)
        @test DifferentialGamesBase.n_players(g) == 3
        @test length(g.initial_state) == 9
        @test length(g.shared_constraints) == 1
        @test length(hallway_game(HallwayParameters(d_comm = 0.5)).shared_constraints) == 2
        case = lookup_case(CaseID(:HallwayPassing, :Nominal, v"1"))
        @test DifferentialGamesBase.n_players(game(case)) == 3
    end

    @testset "warm start" begin
        ws = hallway_warmstart(P)
        @test length(ws.trajectories) == 3
        for tr in ws.trajectories
            @test size(tr.controls) == (2, P.N)
            @test all(0 .<= tr.controls[1, :] .<= P.v_max)
            @test all(abs.(tr.controls[2, :]) .<= P.ω_max)
        end
        # Open-loop unicycle rollout of the plan: on the centre-line through the
        # chokes, goals reached, agents never overlapping (the solver repairs the
        # remaining separation and wall violations)
        paths = map(enumerate(ws.trajectories)) do (i, tr)
            q = copy(hallway_starts_goals(P)[1][i])
            pts = [q[1:2]]
            for k in 1:P.N
                v, ω = tr.controls[:, k]
                θ1 = q[3] + P.dt * ω          # exact unicycle step; RK4 in the solvers agrees to O(dt⁴)
                q = [q[1] + P.dt * v * cos((q[3] + θ1) / 2), q[2] + P.dt * v * sin((q[3] + θ1) / 2), θ1]
                push!(pts, q[1:2])
                @test hallway_halfwidth(P, q[1]) - P.d_wall - abs(q[2] - hallway_centerline(P, q[1])) > -5e-3   # within 5 mm of the inset wall
            end
            @test norm(q[1:2] - hallway_starts_goals(P)[2][i][1:2]) < 0.05
            pts
        end
        for (i, j) in ((1, 2), (1, 3), (2, 3))
            @test minimum(norm(a - b) for (a, b) in zip(paths[i], paths[j])) > 0.05   # a warm start, not a feasible plan: d_safe is 0.17
        end
    end
end

@testset "Racing (Austin)" begin
    tr = austin_track()
    P = RacingParameters()

    @testset "track spline" begin
        @test tr.length ≈ 19.985 atol = 1e-2
        # interpolates its knots and is C¹ (central difference vs analytic derivative)
        @test tr.x(tr.x.t[7]) ≈ tr.x.y[7]
        for s in (1.3, 7.7, 15.2)
            h = 1e-5
            @test (tr.x(s + h) - tr.x(s - h)) / 2h ≈ DifferentialGamesBenchmarks.spline_derivative(tr.x, s) atol = 1e-6
        end
        tx, ty = track_tangent(tr, 4.0)
        @test hypot(tx, ty) ≈ 1
        # a point on the centre-line has zero contouring, lag and lateral error
        ec, el, ey = track_errors(tr, tr.x(4.0), tr.y(4.0), 4.0)
        @test abs(ec) < 1e-12 && abs(el) < 1e-12 && abs(ey) < 1e-12
        # centre-line is parametrised by arc length: unit speed
        @test hypot(DifferentialGamesBenchmarks.spline_derivative(tr.x, 6.0),
                    DifferentialGamesBenchmarks.spline_derivative(tr.y, 6.0)) ≈ 1 atol = 2e-2
    end

    @testset "game" begin
        g = racing_game(tr, P)
        @test DifferentialGamesBase.n_players(g) == 2
        @test length(g.initial_state) == 14
        q1, q2 = racing_starts(tr, P)
        # side-by-side start, clear of each other and inside the track
        @test hypot(q1[4] - q2[4], q1[5] - q2[5]) >= racing_collision_distance(P)
        for q in (q1, q2)
            _, _, ey = track_errors(tr, q[4], q[5], q[7])
            @test abs(ey) < tr.left(0.0)
        end
        @test lookup_case(CaseID(:Racing, :AustinNominal, v"1")) === RACING_AUSTIN_NOMINAL_V1
    end

    @testset "dynamics" begin
        f = bicycle_dynamics(P)
        # straight driving, no input: no lateral acceleration, no yaw acceleration
        dq = f([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0], nothing, 0.0)
        @test abs(dq[2]) < 1e-12
        @test abs(dq[3]) < 1e-12
        @test dq[4] ≈ 2.0 && dq[7] ≈ 2.0       # ẋ = v_x, ṡ = u_s
        # left steer produces positive yaw acceleration
        dq = f([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.2, 2.0], nothing, 0.0)
        @test dq[3] > 0
    end

    @testset "warm start is an on-track reference plan" begin
        ws = racing_warmstart(tr, P)
        f = bicycle_dynamics(P)
        rk4(q, u, h) = (k1 = f(q, u, nothing, 0); k2 = f(q + h / 2 * k1, u, nothing, 0);
                        k3 = f(q + h / 2 * k2, u, nothing, 0); k4 = f(q + h * k3, u, nothing, 0);
                        q + h / 6 * (k1 + 2k2 + 2k3 + k4))
        paths = map(enumerate(racing_starts(tr, P))) do (i, q)
            pts = [q]
            for k in 1:P.N
                q = rk4(q, ws.trajectories[i].controls[:, k], P.dt)   # one RK4 step per dt, as the solvers do
                push!(pts, q)
            end
            pts
        end
        for pts in paths, q in pts
            _, _, ey = track_errors(tr, q[4], q[5], q[7])
            @test -tr.right(q[7]) < ey < tr.left(q[7])
        end
        @test minimum(hypot(a[4] - b[4], a[5] - b[5]) for (a, b) in zip(paths[1], paths[2])) >=
              racing_collision_distance(P) - 5e-3
        @test paths[1][end][7] > 5        # progressed more than 5 m
    end
end

@testset "FALCON suite: Austin benchmark initial conditions, hairpin, hallway with communication" begin
    ics = austin_benchmark_initial_conditions()
    @test size(ics.q0_1) == (50, 7) && size(ics.q0_2) == (50, 7)
    # the Python DG-SQP results stored with them are the DG-SQP row of the FALCON paper's table
    @test count(ics.py_converged) / 50 ≈ 0.72
    @test sum(ics.py_iters[ics.py_converged]) / count(ics.py_converged) ≈ 422.8 atol = 0.05
    tr = austin_track()
    for i in (1, 17, 50)
        g = racing_game_from_ic(i)
        @test g.initial_state ≈ vcat(ics.q0_1[i, :], ics.q0_2[i, :])
        # the initial positions lie on the track
        for q in (ics.q0_1[i, :], ics.q0_2[i, :])
            _, _, ey = track_errors(tr, q[4], q[5], q[7])
            @test -tr.right(q[7]) < ey < tr.left(q[7])
        end
    end
    @test_throws AssertionError racing_game_from_ic(51)

    h = austin_hairpin_track()
    @test h.length ≈ 29.49 atol = 1e-2
    case = lookup_case(CaseID(:Racing, :AustinHairpin, v"1"))
    @test DifferentialGamesBase.n_players(game(case)) == 2

    comm = lookup_case(CaseID(:HallwayPassing, :Communication, v"1"))
    gc = game(comm)
    @test length(gc.shared_constraints) == 2                 # collision and communication range
end
