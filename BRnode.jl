using InfiniteOpt, MathOptAI, Flux, Ipopt, DelimitedFiles, NLPModelsIpopt, HSL_jll
using InfiniteExaModels, ExaModels, MadNLPGPU, CUDA, CUDSS
using PythonCall

# FLUX MODEL--------------------------------------------------------------------
nn_template = Flux.Chain(
    Flux.Scale(ones(Float32,6), zeros(Float32,6)),
    Flux.Dense(6 => 10, tanh),
    Flux.Dense(10 => 10, tanh),
    Flux.Dense(10 => 1),
    Flux.Scale(ones(Float32,1), zeros(Float32,1))
)

θ_template, re = Flux.destructure(nn_template)
θ_trained = vec(readdlm(
    joinpath(@__DIR__, "BRnode_params.txt"),
    Float32
))
BRnode = re(θ_trained)

model_file = joinpath(@__DIR__, "BRnode_torch.pt")
BRnode_torch = MathOptAI.PytorchModel(model_file)

backend_map = Dict(
    "Ipopt" => ExaTranscriptionBackend(NLPModelsIpopt.IpoptSolver),
    "MadNLP" => ExaTranscriptionBackend(MadNLPSolver, backend = CUDABackend())
)

wX  = 10.0
wP  = 10.0
wS  = 2.0
wV  = 0.0
wF  = 0.0
wSf = 0.00
dF  = 0.5
dSf = 0.5
Yxs = 0.5
Ypx = 0.2
hp = 100
tol_val = 1e-4
wall_time = 3.5e2

function build_model(backend, formulation, node, device_type)
    if backend == "MOI"
        model = InfiniteModel()
        set_optimizer(
            model, 
            () -> ExaModels.Optimizer(
                MadNLP.madnlp,
                CUDABackend();
                tol = tol_val,
                max_wall_time = wall_time
            )
        )
    elseif backend == "Ipopt"
        model = InfiniteModel(backend_map[backend])
        set_optimizer_attribute(model, "tol", tol_val)
        set_optimizer_attribute(model, "max_wall_time", wall_time)
        set_optimizer_attribute(model, "linear_solver", "ma97")
        set_optimizer_attribute(model, "print_timing_statistics", "yes")
    else
        model = InfiniteModel(backend_map[backend])
        set_optimizer_attribute(model, "tol", tol_val)
        set_optimizer_attribute(model, "max_wall_time", wall_time)
    end

    @infinite_parameter(
        model,
        t ∈ [0, hp],
        num_supports = hp,
        derivative_method = OrthogonalCollocation(3)
    )

    @variable(model, X >= 0.001, Infinite(t), start = 0.22733347232902065)
    @variable(model, P >= 0.0, Infinite(t), start = 0.02589137216209134)
    @variable(model, S >= 0.001, Infinite(t), start = 0.36980928654131157)
    @variable(model, 1.0 <= V <= 40.0, Infinite(t), start = 15.326602045936013)
    @variable(model, 0.001 <= F <= 0.15, Infinite(t), start = 0.050596489070686645)
    @variable(model, 0.5 <= Sf <= 25.0, Infinite(t), start = 24.99976335097624)
    
    # Pass different options/models/devices based on formulation
    if formulation == "GB"
        f, formulation = MathOptAI.add_predictor(
            model,
            node,
            [X, P, S, V, F, Sf];
            gray_box = true,
            device = device_type
        )
    elseif formulation == "RS"
        f, formulation = MathOptAI.add_predictor(
            model,
            node,
            [X, P, S, V, F, Sf];
            reduced_space = true,
            device = device_type
        )
    else
        f, formulation = MathOptAI.add_predictor(
            model,
            node,
            [X, P, S, V, F, Sf]
        )
    end

    if backend == "MOI" # ExaModels.Optimizer doesn't support parameters
        @finite_parameter(model, X0 == 0.1)
        @finite_parameter(model, P0 == 0.0)
        @finite_parameter(model, S0 == 0.1)
        @finite_parameter(model, V0 == 15.0)
        @finite_parameter(model, Psp == 0.3)
        @constraint(model, X(0) == X0)
        @constraint(model, P(0) == P0)
        @constraint(model, S(0) == S0)
        @constraint(model, V(0) == V0)
    else
        @constraint(model, X(0) == 0.1)
        @constraint(model, P(0) == 0.0)
        @constraint(model, S(0) == 0.1)
        @constraint(model, V(0) == 15.0)
        Psp = 0.3
    end

    rg = only(f[1])
    @constraint(model, ∂(X,t) == -F * X / V + rg)
    @constraint(model, ∂(P,t) == -F * P / V + Ypx * rg)
    @constraint(model, ∂(S,t) == F * (Sf - S) / V - rg / Yxs)
    @constraint(model, ∂(V,t) == F)

    @objective(
        model,
        Min,
        ∫(
            wP * ((P-Psp))^2 +
            wF  * (F)^2 +
            wSf * (Sf)^2 +
            dF  * (∂(F,t))^2 +
            dSf * (∂(Sf,t))^2,
            t
        )
    )
    return model
end

# Solve
CUDA.allowscalar(true)
total_time = @elapsed ocp = build_model("Ipopt", "FS", BRnode, "cpu")
# total_time = @elapsed ocp = build_model("MadNLP", "FS", BRnode_torch, "gpu")
total_time += @elapsed optimize!(ocp)
println("Total_time: $total_time")
println("Objective: $(objective_value(ocp))")
println("Status:  $(termination_status(ocp))")