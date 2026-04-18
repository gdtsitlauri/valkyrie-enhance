#include <reshade.hpp>

// Windows-side template for integrating VALKYRIE with ReShade.
// Build this against the ReShade SDK on the target machine.
extern "C" __declspec(dllexport) const char *NAME = "VALKYRIE";
extern "C" __declspec(dllexport) const char *DESCRIPTION = "VALKYRIE ReShade integration template";
