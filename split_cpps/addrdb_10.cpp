std::vector<CAddress> ReadAnchors(const fs::path& anchors_db_path)
{
    std::vector<CAddress> anchors;
    try {
        DeserializeFileDB(anchors_db_path, CAddress::V2_DISK(anchors));
        LogPrintf("Loaded %i addresses from %s\n", anchors.size(), fs::quoted(fs::PathToString(anchors_db_path.filename())));
    } catch (const std::exception&) {
        anchors.clear();
    }

    fs::remove(anchors_db_path);
    return anchors;
}