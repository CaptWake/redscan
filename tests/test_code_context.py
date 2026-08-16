import unittest
from pathlib import Path
from unittest.mock import ANY, patch

from redscan.code_context import build_redump_bundle, build_redump_context
from redump import (
    ExtractedFunction,
    ExtractionResult,
    ExtractorError,
    FileFormat,
    FunctionInfo,
    Operation,
    VirtualAddress,
)


def _result(
    *,
    backend: str | None = None,
    file_format: FileFormat = FileFormat.PE,
    operation: Operation = Operation.DECOMPILE,
    code: str = "return 0;\n",
) -> ExtractionResult:
    function = ExtractedFunction(
        info=FunctionInfo("main", VirtualAddress(0x401000)),
        code=code,
        operation=operation,
    )
    return ExtractionResult(
        binary=Path("/samples/sample.exe"),
        backend=backend or ("dncil" if file_format is FileFormat.DOTNET else "radare2"),
        operation=operation,
        file_format=file_format,
        functions=(function,),
    )


class CodeContextTests(unittest.TestCase):
    @patch("redscan.code_context.extract")
    @patch("redscan.code_context.detect_format", return_value=FileFormat.PE)
    def test_bundle_preserves_function_boundaries(self, _detect, extract):
        first = ExtractedFunction(
            info=FunctionInfo("first", VirtualAddress(0x401000)),
            code="return 1;\n",
            operation=Operation.DECOMPILE,
        )
        second = ExtractedFunction(
            info=FunctionInfo("second", VirtualAddress(0x402000)),
            code="return 2;\n",
            operation=Operation.DECOMPILE,
        )
        extract.return_value = ExtractionResult(
            binary=Path("/samples/sample.exe"),
            backend="radare2",
            operation=Operation.DECOMPILE,
            file_format=FileFormat.PE,
            functions=(first, second),
        )

        bundle = build_redump_bundle(Path("sample.exe"))

        self.assertEqual(2, len(bundle.code_units))
        self.assertIn("first", bundle.code_units[0])
        self.assertNotIn("second", bundle.code_units[0])
        self.assertIn("second", bundle.code_units[1])
        self.assertEqual("".join(bundle.code_units), bundle.observations["output"])

    @patch("redscan.code_context.extract")
    @patch("redscan.code_context.detect_format", return_value=FileFormat.PE)
    def test_native_auto_mode_uses_radare2_and_returns_complete_output(
        self, _detect, extract
    ):
        extract.return_value = _result(code="A" * 200)
        messages = []

        context = build_redump_context(Path("sample.exe"), progress=messages.append)

        extract.assert_called_once_with(
            Path("sample.exe"),
            backend="radare2",
            operation=Operation.DECOMPILE,
            file_format=FileFormat.PE,
            progress=ANY,
        )
        self.assertEqual("ok", context["status"])
        self.assertEqual("radare2", context["backend"])
        self.assertFalse(context["truncated"])
        self.assertEqual(context["total_characters"], context["included_characters"])
        self.assertIn("A" * 200, context["output"])
        self.assertEqual("Detecting a redump backend...", messages[0])

    @patch("redscan.code_context.extract")
    @patch("redscan.code_context.detect_format", return_value=FileFormat.DOTNET)
    def test_dotnet_auto_mode_uses_dncil_disassembly(self, _detect, extract):
        code = "IL_0000: nop\n" * 5_000
        extract.return_value = _result(
            file_format=FileFormat.DOTNET,
            operation=Operation.DISASSEMBLE,
            code=code,
        )

        context = build_redump_context(Path("sample.exe"))

        extract.assert_called_once_with(
            Path("sample.exe"),
            backend="dncil",
            operation=Operation.DISASSEMBLE,
            file_format=FileFormat.DOTNET,
            progress=None,
        )
        self.assertEqual("disassemble", context["operation"])
        self.assertFalse(context["truncated"])
        self.assertGreater(context["included_characters"], 60_000)
        self.assertEqual(context["total_characters"], context["included_characters"])

    @patch("redscan.code_context.extract")
    @patch("redscan.code_context.detect_format", return_value=FileFormat.ELF)
    def test_explicit_ida_backend_defaults_to_decompilation(self, _detect, extract):
        extract.return_value = _result(
            backend="ida",
            file_format=FileFormat.ELF,
        )

        context = build_redump_context(Path("sample.elf"), backend="ida")

        extract.assert_called_once_with(
            Path("sample.elf"),
            backend="ida",
            operation=Operation.DECOMPILE,
            file_format=FileFormat.ELF,
            progress=None,
        )
        self.assertEqual("ida", context["backend"])
        self.assertEqual("decompile", context["operation"])

    @patch(
        "redscan.code_context.detect_format",
        side_effect=ExtractorError("unrecognized file format"),
    )
    def test_extraction_failure_becomes_model_context(self, _detect):
        context = build_redump_context(Path("sample.pdf"))

        self.assertEqual("unavailable", context["status"])
        self.assertEqual("auto", context["requested_backend"])
        self.assertIn("unrecognized file format", context["error"])


if __name__ == "__main__":
    unittest.main()
