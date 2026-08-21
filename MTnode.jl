using InfiniteOpt, MathOptAI, Flux, Ipopt, DelimitedFiles, NLPModelsIpopt, HSL_jll
using InfiniteExaModels, ExaModels, MadNLPGPU, CUDA, CUDSS
using PythonCall

# FLUX MODEL--------------------------------------------------------------------
nn_template =  Flux.Chain(
    Flux.Scale(ones(Float32, 4), zeros(Float32, 4)), 
    Flux.Dense(4 => 10, tanh),
    Flux.Dense(10 => 10, tanh),
    Flux.Dense(10 => 2),
    Flux.Scale(ones(Float32, 2), zeros(Float32, 2))
)
θ_template, re = Flux.destructure(nn_template)
θ_trained = vec(readdlm(joinpath(@__DIR__, "MTnode_params.txt"), Float32))
MTnode = re(θ_trained)

model_file = joinpath(@__DIR__, "MTnode_torch.pt")
MTnode_torch = MathOptAI.PytorchModel(model_file)

backend_map = Dict(
    "Ipopt" => ExaTranscriptionBackend(NLPModelsIpopt.IpoptSolver),
    "MadNLP" => ExaTranscriptionBackend(MadNLPSolver, backend = CUDABackend())
)

w1 = 2.5
w2 = 5
w3 = 0.25
w4 = 0.25
w5 = 15
n = 50
tol_val = 1e-4
wall_time = 3.5e2

function steps(t)
    if t <= n/2
        return (10, 340) 
    else
        return (5, 322) 
    end
end

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
        t ∈ [0, n],
        num_supports = n,
        derivative_method = OrthogonalCollocation(3)
    )

    @variable(model, 0.001 <= V, Infinite(t), start = 10.928454611007794)
    @variable(model, T <= 370, Infinite(t), start = 353.92169778277344)
    @variable(model, 0.015 <= Fc <= 2.0, Infinite(t), start = 0.2870844314382967)
    @variable(model, 0 <= z <= 1, Infinite(t), start = 0.8807340904682742)

    # Pass different options/models/devices based on formulation
    if formulation == "GB"
        f, formulation = MathOptAI.add_predictor(
            model,
            node,
            [V, T, Fc, z];
            gray_box = true,
            device = device_type
        )
    elseif formulation == "RS"
        f, formulation = MathOptAI.add_predictor(
            model,
            node,
            [V, T, Fc, z];
            reduced_space = true,
            device = device_type
        )
    else
        f, formulation = MathOptAI.add_predictor(
            model,
            node,
            [V, T, Fc, z]
        )
    end

    @constraint(model, ∂(V, t) == only(f[1]))
    @constraint(model, ∂(T, t) == only(f[2]))
    @constraint(model, V(0) == 15)
    @constraint(model, T(0) == 360)

    T_sp = parameter_function(t, name = "T_sp") do tau
        steps(tau)[2]
    end

    V_sp = parameter_function(t, name = "V_sp") do tau
        steps(tau)[1]
    end

    @objective(
        model,
        Min,
        ∫(
            w1*((V - V_sp)/4.1)^2 +
            w2*((T-T_sp)/16.2)^2 +
            w3*(∂(V, t))^2 +
            w4*(∂(T, t))^2 +
            w5*(∂(Fc, t))^2 +
            w5*(∂(z, t))^2,
            t
        )
    )
    return model
end

# Solve
CUDA.allowscalar(true)
total_time = @elapsed ocp = build_model("Ipopt", "FS", MTnode, "cpu")
# total_time = @elapsed ocp = build_model("MadNLP", "FS", MTnode_torch, "gpu")
total_time += @elapsed optimize!(ocp)
status = termination_status(ocp)
println("Total_time: $total_time")
println("Objective: $(objective_value(ocp))")
println("Status:  $(termination_status(ocp))")