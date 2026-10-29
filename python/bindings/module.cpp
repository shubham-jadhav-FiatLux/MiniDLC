// The compiled half of the minidlc package: compile(ir_text, target, options) -> Executable. The graph crosses the
// boundary as IR text; NumPy arrays cross as borrowed buffers (inputs) and new arrays (outputs).
#include "dlc/compiler/Compiler.h"
#include "dlc/ir/Error.h"
#include <cstring>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using dlc::runtime::Executable;

namespace {

dlc::DataType dtypeOf(nb::dlpack::dtype dtype) {
    if (dtype == nb::dtype<float>()) return dlc::DataType::Float32;
    if (dtype == nb::dtype<int32_t>()) return dlc::DataType::Int32;
    throw nb::type_error("arrays must be float32 or int32");
}

nb::object toNumpy(dlc::runtime::HostTensor t) {
    auto* data = new std::vector<char>(std::move(t.data));
    nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<char>*>(p); });
    std::vector<size_t> shape(t.shape.begin(), t.shape.end());
    if (t.dtype == dlc::DataType::Float32) {
        return nb::cast(nb::ndarray<nb::numpy, float>(data->data(), shape.size(), shape.data(), owner));
    }
    return nb::cast(nb::ndarray<nb::numpy, int32_t>(data->data(), shape.size(), shape.data(), owner));
}

std::string roleName(dlc::ArgRole role) { return dlc::argRoleName(role); }

nb::dict describe(const dlc::runtime::Executable& e) {
    nb::list args, results, launches;
    for (const auto& a : e.plan().args) {
        nb::dict d;
        d["name"] = a.name;
        d["role"] = roleName(a.role);
        d["dtype"] = dlc::dataTypeToString(a.dtype);
        d["shape"] = nb::tuple(nb::cast(std::vector<int64_t>(a.shape.begin(), a.shape.end())));
        args.append(d);
    }
    for (const auto& r : e.plan().results) {
        nb::dict d;
        d["kind"] = dlc::resultKindName(r.kind);
        d["arg"] = r.arg;
        d["dtype"] = dlc::dataTypeToString(r.dtype);
        d["shape"] = nb::tuple(nb::cast(std::vector<int64_t>(r.shape.begin(), r.shape.end())));
        results.append(d);
    }
    for (const auto& l : e.plan().launches) {
        nb::dict d;
        d["name"] = l.name;
        d["parallel_extent"] = l.parallelExtent;
        d["flops"] = l.flops;
        d["bytes"] = l.bytes;
        launches.append(d);
    }
    nb::dict out;
    out["function"] = e.plan().function;
    out["args"] = args;
    out["results"] = results;
    out["launches"] = launches;
    out["persistent_bytes"] = e.plan().persistentBytes;
    out["transient_bytes"] = e.plan().transientBytes;
    return out;
}

dlc::compiler::CompileOptions parseOptions(const std::string& target, nb::dict options) {
    dlc::compiler::CompileOptions opts;
    opts.target = target;
    if (options.contains("threads")) opts.threads = nb::cast<size_t>(options["threads"]);
    if (options.contains("fuse")) opts.fuse = nb::cast<bool>(options["fuse"]);
    if (options.contains("check_all")) opts.lowering.checkEveryAccess = nb::cast<bool>(options["check_all"]);
    if (options.contains("matmul_schedule")) {
        std::string schedule = nb::cast<std::string>(options["matmul_schedule"]);
        if (schedule != "tiled" && schedule != "naive") throw nb::value_error("matmul_schedule must be 'tiled' or 'naive'");
        opts.lowering.naiveMatmul = schedule == "naive";
    }
    if (options.contains("optimizer") && !options["optimizer"].is_none()) {
        dlc::OptimizerConfig config;
        std::string kind = nb::cast<std::string>(options["optimizer"]);
        if (kind == "sgd") config.kind = dlc::OptimizerConfig::Kind::SGD;
        else if (kind == "adam") config.kind = dlc::OptimizerConfig::Kind::Adam;
        else throw nb::value_error("optimizer must be 'adam' or 'sgd'");
        auto number = [&](const char* key, double& field) {
            if (options.contains(key)) field = nb::cast<double>(options[key]);
        };
        number("lr", config.lr);
        number("momentum", config.momentum);
        number("beta1", config.beta1);
        number("beta2", config.beta2);
        number("eps", config.eps);
        opts.optimizer = config;
    }
    return opts;
}

nb::dict dumpsDict(const dlc::compiler::CompileDumps& dumps) {
    nb::dict d;
    d["graph"] = dumps.graphIR;
    d["loopir"] = dumps.loopIR;
    d["llvm"] = dumps.llvmIR;
    d["llvm_opt"] = dumps.llvmIROpt;
    d["cost"] = dumps.costReport;
    d["memory"] = dumps.memoryReport;
    return d;
}

} // namespace

NB_MODULE(_minidlc, m) {
    m.doc() = "MiniDLC compiler: compile(ir_text, target, options) -> (Executable, dumps)";
    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const dlc::IRError& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        }
    });

    nb::class_<Executable>(m, "Executable")
        .def("run",
             [](Executable& e, nb::dict inputs) {
                 std::unordered_map<std::string, dlc::runtime::TensorView> views;
                 std::vector<nb::ndarray<nb::c_contig, nb::device::cpu>> keep;
                 for (auto [key, value] : inputs) {
                     auto arr = nb::cast<nb::ndarray<nb::c_contig, nb::device::cpu>>(value);
                     dlc::Shape shape;
                     for (size_t d = 0; d < arr.ndim(); ++d) shape.push_back(static_cast<int64_t>(arr.shape(d)));
                     views[nb::cast<std::string>(key)] = dlc::runtime::TensorView{dtypeOf(arr.dtype()), shape, arr.data(),
                                                                                  arr.nbytes()};
                     keep.push_back(arr);
                 }
                 std::vector<dlc::runtime::HostTensor> out;
                 {
                     nb::gil_scoped_release release;
                     out = e.run(views);
                 }
                 nb::list results;
                 for (auto& t : out) results.append(toNumpy(std::move(t)));
                 return results;
             },
             "Runs one step. `inputs` maps input names to NumPy arrays; returns the results that are not donated.")
        .def("set",
             [](Executable& e, const std::string& name, nb::ndarray<nb::c_contig, nb::device::cpu> arr) {
                 e.setArg(name, arr.data(), arr.nbytes());
             },
             "Copies a param, const or state argument into the executable.")
        .def("get", [](const Executable& e, const std::string& name) { return toNumpy(e.getArg(name)); },
             "A copy of a param, const or state argument.")
        .def("reset_state", &Executable::resetState, "Resets every state argument to its initial value.")
        .def("set_profiling", &Executable::setProfiling)
        .def("clear_profile", &Executable::clearProfile)
        .def("profile",
             [](const Executable& e) {
                 nb::list out;
                 for (const auto& p : e.profile()) {
                     nb::dict d;
                     d["name"] = p.name;
                     d["calls"] = p.calls;
                     d["seconds"] = p.seconds;
                     d["flops"] = p.flops;
                     d["bytes"] = p.bytes;
                     out.append(d);
                 }
                 return out;
             })
        .def_prop_ro("info", &describe);

    m.def(
        "compile",
        [](const std::string& ir, const std::string& target, nb::dict options) {
            dlc::compiler::CompileOptions opts = parseOptions(target, options);
            dlc::compiler::CompileDumps dumps;
            std::unique_ptr<Executable> exe;
            {
                nb::gil_scoped_release release;
                exe = dlc::compiler::compile(ir, opts, &dumps);
            }
            return nb::make_tuple(std::move(exe), dumpsDict(dumps));
        },
        nb::arg("ir"), nb::arg("target") = "cpu", nb::arg("options") = nb::dict(),
        "Compiles the first function of `ir` for `target` ('cpu', 'interp' or 'gpu-sim'). Options: optimizer ('adam' or 'sgd'), "
        "lr, momentum, beta1, beta2, eps, threads; ablations: fuse (bool), check_all (bool), matmul_schedule "
        "('tiled' or 'naive'). Returns (Executable, dumps).");

    using dlc::compiler::GpuArtifact;
    nb::class_<GpuArtifact>(m, "GpuArtifact")
        .def_prop_ro("plan_json", [](const GpuArtifact& a) { return dlc::runtime::toJson(a.plan); })
        .def_prop_ro("ptx", [](const GpuArtifact& a) { return a.ptx; })
        .def_prop_ro("arch", [](const GpuArtifact& a) { return a.arch; })
        .def_prop_ro("kernel_names",
                     [](const GpuArtifact& a) {
                         std::vector<std::string> names;
                         for (const auto& k : a.kernels) names.push_back(k.name);
                         return names;
                     })
        .def(
            "simulate",
            [](const GpuArtifact& a, int kernel, const std::vector<uintptr_t>& pointers) {
                std::vector<void*> args;
                for (uintptr_t p : pointers) args.push_back(reinterpret_cast<void*>(p));
                if (kernel < 0 || static_cast<size_t>(kernel) >= a.kernels.size()) throw nb::index_error("no such kernel");
                if (args.size() != a.kernels[kernel].globalBuffers().size()) throw nb::value_error("wrong number of buffers");
                nb::gil_scoped_release release;
                dlc::compiler::simulateLaunch(a, kernel, args);
            },
            nb::arg("kernel"), nb::arg("pointers"),
            "Runs kernel `kernel` with the lockstep GPU interpreter; `pointers` are host addresses of its buffers.");

    m.def(
        "compile_gpu",
        [](const std::string& ir, nb::dict options, bool ptx, const std::string& arch) {
            dlc::compiler::CompileOptions opts = parseOptions("cuda", options);
            dlc::compiler::CompileDumps dumps;
            GpuArtifact artifact;
            {
                nb::gil_scoped_release release;
                artifact = dlc::compiler::compileGpu(ir, opts, &dumps, ptx, arch);
            }
            return nb::make_tuple(std::move(artifact), dumpsDict(dumps));
        },
        nb::arg("ir"), nb::arg("options") = nb::dict(), nb::arg("ptx") = true, nb::arg("arch") = "sm_75",
        "Compiles the first function of `ir` with the GPU schedules: (GpuArtifact, dumps). The artifact holds the plan "
        "(as JSON) and, with ptx=True, the PTX of every kernel for `arch`.");
}
