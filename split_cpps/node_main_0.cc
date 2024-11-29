void ExitIfContainsDisallowedFlags(const std::vector<std::string>& argv) {
  // Options that are unilaterally disallowed.
  static constexpr auto disallowed = base::MakeFixedFlatSet<std::string_view>({
      "--enable-fips",
      "--force-fips",
      "--openssl-config",
      "--use-bundled-ca",
      "--use-openssl-ca",
  });

  for (const auto& arg : argv) {
    const auto key = std::string_view{arg}.substr(0, arg.find('='));
    if (disallowed.contains(key)) {
      LOG(ERROR) << "The Node.js cli flag " << key
                 << " is not supported in Electron";
      // Node.js returns 9 from ProcessGlobalArgs for any errors encountered
      // when setting up cli flags and env vars. Since we're outlawing these
      // flags (making them errors) exit with the same error code for
      // consistency.
      exit(9);
    }
  }
}