bool IsSandboxEnabled(base::CommandLine* command_line) {
  return command_line->HasSwitch(switches::kEnableSandbox) ||
         !command_line->HasSwitch(sandbox::policy::switches::kNoSandbox);
}