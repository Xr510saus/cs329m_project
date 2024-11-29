bool CBanDB::Read(banmap_t& banSet)
{
    if (fs::exists(m_banlist_dat)) {
        LogPrintf("banlist.dat ignored because it can only be read by " CLIENT_NAME " version 22.x. Remove %s to silence this warning.\n", fs::quoted(fs::PathToString(m_banlist_dat)));
    }
    // If the JSON banlist does not exist, then recreate it
    if (!fs::exists(m_banlist_json)) {
        return false;
    }

    std::map<std::string, common::SettingsValue> settings;
    std::vector<std::string> errors;

    if (!common::ReadSettings(m_banlist_json, settings, errors)) {
        for (const auto& err : errors) {
            LogPrintf("Cannot load banlist %s: %s\n", fs::PathToString(m_banlist_json), err);
        }
        return false;
    }

    try {
        BanMapFromJson(settings[JSON_KEY], banSet);
    } catch (const std::runtime_error& e) {
        LogPrintf("Cannot parse banlist %s: %s\n", fs::PathToString(m_banlist_json), e.what());
        return false;
    }

    return true;
}