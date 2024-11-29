bool CBanDB::Write(const banmap_t& banSet)
{
    std::vector<std::string> errors;
    if (common::WriteSettings(m_banlist_json, {{JSON_KEY, BanMapToJson(banSet)}}, errors)) {
        return true;
    }

    for (const auto& err : errors) {
        LogError("%s\n", err);
    }
    return false;
}