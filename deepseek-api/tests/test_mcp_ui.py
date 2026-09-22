import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PAGE = ROOT / "static" / "index.html"
SCRIPT = ROOT / "static" / "mcp-controls.js"
README = ROOT / "README.md"


class McpUiTests(unittest.TestCase):
    def test_fixed_deepwiki_tab_has_explicit_connect_action(self):
        page = PAGE.read_text(encoding="utf-8")

        for fragment in (
            'id="view-mcp"',
            '>MCP<',
            'id="mcp-view" hidden',
            'id="mcp-connect-form"',
            'id="mcp-connect"',
            'id="mcp-status"',
            'id="mcp-tools"',
            '<script src="/static/mcp-controls.js"></script>',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, page)
        self.assertNotIn('id="mcp-url"', page)

    def test_controls_post_empty_request_and_render_untrusted_tool_text_safely(self):
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('form.addEventListener("submit", async (event) =>', source)
        self.assertIn('fetch("/api/mcp/tools", {method:"POST"})', source)
        self.assertNotIn('body:', source)
        self.assertNotIn('innerHTML', source)
        self.assertIn('name.textContent=tool.name', source)
        self.assertIn('description.textContent=tool.description', source)
        self.assertIn('body.error', source)

    def test_readme_documents_day_16_and_python_310(self):
        readme = README.read_text(encoding="utf-8")

        self.assertIn('[День 16 — подключение MCP]', readme)
        self.assertIn('## День 16 — подключение MCP', readme)
        self.assertIn('Python 3.10', readme)
        self.assertIn('https://mcp.deepwiki.com/mcp', readme)


if __name__ == "__main__":
    unittest.main()
