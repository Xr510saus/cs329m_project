bool CheckCommandLineArguments(const base::CommandLine::StringVector& argv) {
  bool block_args = false;
  for (const auto& arg : argv) {
    if (arg == DashDash)
      break;
    if (block_args)
      return false;
    if (IsUrlArg(arg))
      block_args = true;
  }
  return true;
}