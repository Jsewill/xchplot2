// Exercise the foreground dashboard against a private Windows console.
#include "host/GroupDashboard.hpp"
#include <fcntl.h>
#include <stdexcept>
#include <utility>

namespace
{
void check(bool ok, char const* message)
{ if (!ok) throw std::runtime_error(message); }

// Preserve the exact stderr bytes while forwarding them to the console.
class ConsoleOutput : public std::streambuf
{
    std::streambuf* destination_ = std::cerr.rdbuf(this);
    std::string bytes_;
    std::streamsize xsputn(char const* data, std::streamsize count) override
    {
        bytes_.append(data, size_t(count));
        return destination_->sputn(data, count);
    }
    int_type overflow(int_type c) override
    {
        if (traits_type::eq_int_type(c, traits_type::eof())) return traits_type::not_eof(c);
        bytes_ += traits_type::to_char_type(c);
        return destination_->sputc(traits_type::to_char_type(c));
    }
    int sync() override { return destination_->pubsync(); }
public:
    ~ConsoleOutput() { std::cerr.rdbuf(destination_); }
    std::string take() { return std::exchange(bytes_, {}); }
};

void key(HANDLE input, WORD key_code, wchar_t character, WORD repeat, DWORD modifiers)
{
    INPUT_RECORD events[2]{};
    auto& down = events[0].Event.KeyEvent;
    events[0].EventType = KEY_EVENT; down.bKeyDown = TRUE; down.wRepeatCount = repeat;
    down.wVirtualKeyCode = key_code;
    down.wVirtualScanCode = WORD(::MapVirtualKeyW(key_code, MAPVK_VK_TO_VSC));
    down.uChar.UnicodeChar = character; down.dwControlKeyState = modifiers;
    events[1] = events[0]; events[1].Event.KeyEvent.bKeyDown = FALSE;
    DWORD sent = 0;
    check(::WriteConsoleInputW(input, events, 2, &sent) && sent == 2, "failed to inject console key records");
}

void keys(HANDLE input, std::wstring_view characters)
{
    for (wchar_t c : characters)
    {
        auto const code = ::VkKeyScanW(c);
        check(code != -1, "test console keyboard cannot represent the requested key");
        key(input, LOBYTE(code), c, 1, ((HIBYTE(code) & 1) ? SHIFT_PRESSED : 0) |
            ((HIBYTE(code) & 2) ? LEFT_CTRL_PRESSED : 0) | ((HIBYTE(code) & 4) ? LEFT_ALT_PRESSED : 0));
    }
}

void foreground(bool private_console)
{
    // CTest may inherit a terminal belonging to its shell. Give this process
    // its own console before changing modes or inserting keyboard records.
    DWORD processes[2]{};
    if (!private_console && ::GetConsoleProcessList(processes, 2) != 1)
    {
        ::FreeConsole();
        check(::AllocConsole(), "could not allocate a private Windows test console");
    }
    check(std::freopen("CONIN$", "r+", stdin) && std::freopen("CONOUT$", "r+", stderr), "cannot open the private console streams");
    auto native = [](FILE* file) { return reinterpret_cast<HANDLE>(::_get_osfhandle(::_fileno(file))); };
    auto input = native(stdin), output = native(stderr);
    check(::SetStdHandle(STD_INPUT_HANDLE, input) && ::SetStdHandle(STD_ERROR_HANDLE, output), "could not bind the console streams");
    DWORD initial_input = 0, initial_output = 0;
    check(::GetConsoleMode(input, &initial_input) && ::GetConsoleMode(output, &initial_output), "could not inspect initial console modes");
    initial_output &= ~DWORD(ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    check(::SetConsoleMode(output, initial_output), "could not establish the initial console mode");
    CONSOLE_SCREEN_BUFFER_INFO screen{};
    CONSOLE_CURSOR_INFO cursor{};
    check(::GetConsoleScreenBufferInfo(output, &screen) && ::GetConsoleCursorInfo(output, &cursor), "cannot inspect the original screen");
    COORD const top{screen.srWindow.Left, screen.srWindow.Top};
    DWORD cells = 0;
    check(::WriteConsoleOutputCharacterW(output, L"~", 1, top, &cells) && cells == 1, "cannot mark the original screen");
    auto restored = [&]
    {
        // In a pseudoconsole the display host owns the alternate screen.
        // Check its enter/leave stream below and these Win32-owned modes here.
        DWORD in = 0, out = 0; CONSOLE_CURSOR_INFO visible{};
        return ::GetConsoleMode(input, &in) && in == initial_input && ::GetConsoleMode(output, &out) && out == initial_output &&
            ::GetConsoleCursorInfo(output, &visible) && visible.bVisible == cursor.bVisible;
    };

    int channel[2];
    check(!::_pipe(channel, 65536, _O_BINARY), "could not create machine output pipe");
    check(!::_dup2(channel[1], ::_fileno(stdout)) && ::SetStdHandle(STD_OUTPUT_HANDLE, native(stdout)), "could not redirect machine output");
    ::_close(channel[1]); ::_putenv("TERM=xterm");
    std::cout << "{\"durable\":1}\n" << std::flush;
    group_grow::DashboardFrame frame;
    frame.title = "Private console";
    frame.sharing = group_grow::DashboardFrame::Sharing::enrolled;
    frame.add(group_grow::DashboardFrame::workers, "Online helper");
    frame.add(group_grow::DashboardFrame::workers, "Local CPU");
    frame.add(group_grow::DashboardFrame::workers, "Local GPU");
    ConsoleOutput console;
    {
        group_grow::DashboardTerminal dashboard(true);
        check(dashboard.active() && dashboard.interactive(), "foreground dashboard did not claim its terminal");
        DWORD active_input = 0;
        check(::GetConsoleMode(input, &active_input) && !(active_input & (ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT | ENABLE_VIRTUAL_TERMINAL_INPUT)) &&
            (active_input & ENABLE_PROCESSED_INPUT) == (initial_input & ENABLE_PROCESSED_INPUT),
            "foreground keyboard mode echoes commands, buffers lines, or changes Ctrl-C handling");
        dashboard.draw(frame);
        check(console.take().find("\033[?1049h") != std::string::npos, "foreground dashboard did not open its screen");
        INPUT_RECORD resize{}; resize.EventType = WINDOW_BUFFER_SIZE_EVENT;
        resize.Event.WindowBufferSizeEvent.dwSize = screen.dwSize;
        DWORD sent = 0;
        check(::WriteConsoleInputW(input, &resize, 1, &sent) && sent == 1 && !dashboard.keys(),
            "a console resize waited for a keystroke or changed keyboard navigation");
        auto press = [&](std::wstring_view input_keys, char const* expected)
        {
            keys(input, input_keys);
            check(dashboard.keys(), "dashboard did not receive a console key");
            dashboard.draw(frame);
            check(console.take().find(expected) != std::string::npos, expected);
        };
        press(L"?", "Dashboard help");
        press(L"a", "xchplot2 compute setup helper.invite");
        press(L"3", "Online helper");
        key(input, VK_DOWN, 0, 2, ENHANCED_KEY);
        press(L"", "> Local GPU");
        press(L"/absent\r", "No matching rows");
        press(L"q", "\033[?25h\033[?1049l");
        check(!dashboard.active() && restored(), "q did not restore the console and return to foreground work");
    }
    {
        group_grow::DashboardTerminal dashboard(true);
        dashboard.draw(frame);
        (void)console.take();
    }
    check(console.take().find("\033[?25h\033[?1049l") != std::string::npos && restored(),
        "dashboard destruction did not restore cursor, screen and console modes");
    std::cout << "{\"durable\":2}\n" << std::flush;
    ::_close(::_fileno(stdout));
    char machine[64];
    int const count = ::_read(channel[0], machine, sizeof(machine));
    check(count > 0 && std::string_view(machine, size_t(count)) == "{\"durable\":1}\n{\"durable\":2}\n" &&
        ::_read(channel[0], machine, 1) == 0, "dashboard controls contaminated machine-readable stdout");
    ::_close(channel[0]);
}
}

int main(int argc, char** argv)
{
    auto report = ::_fdopen(::_dup(::_fileno(stdout)), "w");
    try
    {
        bool const private_console = argc == 2 && std::string_view(argv[1]) == "--private-console";
        check(argc == 1 || private_console, "supply --private-console only inside an isolated test terminal");
        foreground(private_console);
        if (report) std::fputs("Windows foreground console passed\n", report);
    }
    catch (std::exception const& error)
    {
        if (report) { std::fprintf(report, "%s (Win32 %lu)\n", error.what(), ::GetLastError()); std::fclose(report); }
        return 1;
    }
    if (report) std::fclose(report);
}
