from __future__ import annotations

from pathlib import Path


def generate_reshade_bundle(output_dir: str) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    addon = root / "VALKYRIE_Addon_Template.cpp"
    ini = root / "VALKYRIE.ini"
    readme = root / "README.md"
    addon.write_text(
        "\n".join(
            [
                "#include <reshade.hpp>",
                "",
                "// Windows-side template for integrating VALKYRIE with ReShade.",
                "// Build this against the ReShade SDK on the target machine.",
                "extern \"C\" __declspec(dllexport) const char *NAME = \"VALKYRIE\";",
                "extern \"C\" __declspec(dllexport) const char *DESCRIPTION = \"VALKYRIE ReShade integration template\";",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ini.write_text(
        "\n".join(
            [
                "[VALKYRIE]",
                "Enabled=1",
                "Transport=shared_memory",
                "PipeName=VALKYRIE_RT",
                "TargetFPS=60",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    readme.write_text(
        "\n".join(
            [
                "# ReShade Integration",
                "",
                "This bundle is a Windows-oriented starter for connecting a ReShade add-on to the VALKYRIE runtime.",
                "Build the add-on against the ReShade SDK on the target Windows 11 machine, then bridge frames to the VALKYRIE process through shared memory or a named pipe.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"addon": str(addon), "ini": str(ini), "readme": str(readme)}
