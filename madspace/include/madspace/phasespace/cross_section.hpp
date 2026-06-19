#pragma once

#include "madspace/phasespace/matrix_element.hpp"
#include "madspace/phasespace/pdf.hpp"
#include "madspace/phasespace/scale.hpp"

namespace madspace {

class DifferentialCrossSection : public FunctionGenerator {
public:
    struct CachedPdf {};

    DifferentialCrossSection(
        const MatrixElement& matrix_element,
        double cm_energy,
        const RunningCoupling& running_coupling,
        const EnergyScale& energy_scale,
        const nested_vector2<me_int_t>& pid_options = {},
        const std::variant<std::monostate, PdfGrid, CachedPdf>& pdf1 = std::monostate{},
        const std::variant<std::monostate, PdfGrid, CachedPdf>& pdf2 = std::monostate{},
        bool has_mirror = false,
        bool input_momentum_fraction = true
    );

    const nested_vector2<me_int_t>& pid_options() const { return _pid_options; }
    bool has_mirror() const { return _has_mirror; }
    bool has_pdf(std::size_t pdf_index) const { return _has_pdf.at(pdf_index); }
    const MatrixElement& matrix_element() const { return _matrix_element; }
    const EnergyScale& energy_scale() const { return _energy_scale; }
    const RunningCoupling& running_coupling() const { return _running_coupling; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    nested_vector2<me_int_t> _pid_options;
    MatrixElement _matrix_element;
    std::array<std::optional<PartonDensity>, 2> _pdfs;
    std::array<bool, 2> _has_pdf;
    RunningCoupling _running_coupling;
    double _e_cm;
    EnergyScale _energy_scale;
    bool _has_mirror;
    bool _input_momentum_fraction;
};

} // namespace madspace
