bool IsUrlArg(const base::CommandLine::StringViewType arg) {
  const auto scheme_end = arg.find(':');
  if (scheme_end == base::CommandLine::StringViewType::npos)
    return false;

  const auto& c_locale = std::locale::classic();
  const auto isspace = [&](auto ch) { return std::isspace(ch, c_locale); };
  const auto scheme = arg.substr(0U, scheme_end);
  return std::size(scheme) > 1U && std::isalpha(scheme.front(), c_locale) &&
         std::ranges::none_of(scheme, isspace);
}