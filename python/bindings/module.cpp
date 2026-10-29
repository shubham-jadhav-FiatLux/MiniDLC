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
            dlc::compiler::CompileOptions opts;
            opts.target = target;
            if (options.contains("threads")) opts.threads = nb::cast<size_t>(options["threads"]);
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
            dlc::compiler::CompileDumps dumps;
            std::unique_ptr<Executable> exe;
            {
                nb::gil_scoped_release release;
                exe = dlc::compiler::compile(ir, opts, &dumps);
            }
            nb::dict d;
            d["graph"] = dumps.graphIR;
            d["loopir"] = dumps.loopIR;
            d["llvm"] = dumps.llvmIR;
            d["llvm_opt"] = dumps.llvmIROpt;
            d["cost"] = dumps.costReport;
            d["memory"] = dumps.memoryReport;
            return nb::make_tuple(std::move(exe), d);
        },
        nb::arg("ir"), nb::arg("target") = "cpu", nb::arg("options") = nb::dict(),
        "Compiles the first function of `ir` for `target` ('cpu' or 'interp'). Options: optimizer ('adam' or 'sgd'), "
        "lr, momentum, beta1, beta2, eps, threads. Returns (Executable, dumps).");
}
