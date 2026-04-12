import json

from playwright.async_api import async_playwright

STEP_SCHEMA = [
    "navigate", "click", "fill", "type_text", "clear", "press_key",
    "scroll", "scroll_to_element", "screenshot", "get_text", "get_attribute",
    "get_url", "get_title", "hover", "double_click", "right_click",
    "wait_for_selector", "wait_text", "select_option", "check", "uncheck",
    "upload_file", "extract_links", "extract_table", "assert_text", "assert_url",
    "save_cookies", "load_cookies", "new_tab", "switch_tab", "close_tab", "list_tabs",
]


class BrowserAgent:
    def __init__(self, headless=True):
        self.headless = headless
        self._playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self._tabs = []

    async def start(self):
        self._playwright = await async_playwright().start()
        self.browser = await self._playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()

    async def stop(self):
        await self.browser.close()
        await self._playwright.stop()

    async def execute_step(self, step: dict) -> dict:
        try:
            result = await self._dispatch(step)
            if result is None:
                return {"success": True}
            return {"success": True, **result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _dispatch(self, step: dict) -> dict | None:
        action = step.get("action")
        page = self.page

        if action == "navigate":
            await page.goto(step["url"])
        elif action == "click":
            await page.click(step["selector"])
        elif action == "fill":
            await page.fill(step["selector"], step["value"])
        elif action == "type_text":
            await page.type(step["selector"], step["text"])
        elif action == "clear":
            await page.fill(step["selector"], "")
        elif action == "press_key":
            await page.press(step.get("selector", "body"), step["key"])
        elif action == "scroll":
            await page.evaluate(
                "([x, y]) => window.scrollBy(x, y)",
                [step.get("x", 0), step.get("y", 100)],
            )
        elif action == "scroll_to_element":
            await page.eval_on_selector(step["selector"], "el => el.scrollIntoView()")
        elif action == "screenshot":
            await page.screenshot(path=step.get("path", "screenshot.png"))
        elif action == "get_text":
            return {"text": await page.inner_text(step["selector"])}
        elif action == "get_attribute":
            return {"attribute": await page.get_attribute(step["selector"], step["name"])}
        elif action == "get_url":
            return {"url": page.url}
        elif action == "get_title":
            return {"title": await page.title()}
        elif action == "hover":
            await page.hover(step["selector"])
        elif action == "double_click":
            await page.dblclick(step["selector"])
        elif action == "right_click":
            await page.click(step["selector"], button="right")
        elif action == "wait_for_selector":
            await page.wait_for_selector(step["selector"])
        elif action == "wait_text":
            await page.wait_for_selector(f"text={step['text']}")
        elif action == "select_option":
            await page.select_option(step["selector"], step["value"])
        elif action == "check":
            await page.check(step["selector"])
        elif action == "uncheck":
            await page.uncheck(step["selector"])
        elif action == "upload_file":
            await page.set_input_files(step["selector"], step["path"])
        elif action == "extract_links":
            links = await page.evaluate(
                "Array.from(document.querySelectorAll('a')).map(a => a.href)"
            )
            return {"links": links}
        elif action == "extract_table":
            rows = await page.evaluate(
                """() => {
                    const table = document.querySelector('table');
                    if (!table) return [];
                    return Array.from(table.rows).map(r =>
                        Array.from(r.cells).map(c => c.innerText)
                    );
                }"""
            )
            return {"rows": rows}
        elif action == "assert_text":
            text = await page.inner_text(step["selector"])
            if step["expected"] not in text:
                raise AssertionError(
                    f"Expected '{step['expected']}' in text '{text}'"
                )
        elif action == "assert_url":
            if page.url != step["expected"]:
                raise AssertionError(
                    f"Expected URL '{step['expected']}', got '{page.url}'"
                )
        elif action == "save_cookies":
            cookies = await self.context.cookies()
            with open(step["path"], "w") as f:
                json.dump(cookies, f)
        elif action == "load_cookies":
            with open(step["path"]) as f:
                cookies = json.load(f)
            await self.context.add_cookies(cookies)
        elif action == "new_tab":
            new_page = await self.context.new_page()
            self._tabs.append(new_page)
        elif action == "switch_tab":
            self.page = self._tabs[step["index"]]
        elif action == "close_tab":
            await self._tabs[step["index"]].close()
        elif action == "list_tabs":
            return {"tabs": [p.url for p in self._tabs]}
        else:
            raise ValueError(f"Unknown action: {action}")
        return None

    async def run_task(self, steps: list[dict]) -> list[dict]:
        results = []
        for step in steps:
            result = await self.execute_step(step)
            results.append(result)
        return results
