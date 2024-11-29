bool SerializeFileDB(const std::string& prefix, const fs::path& path, const Data& data)
{
    // Generate random temporary filename
    const uint16_t randv{FastRandomContext().rand<uint16_t>()};
    std::string tmpfn = strprintf("%s.%04x", prefix, randv);

    // open temp output file
    fs::path pathTmp = gArgs.GetDataDirNet() / fs::u8path(tmpfn);
    FILE *file = fsbridge::fopen(pathTmp, "wb");
    AutoFile fileout{file};
    if (fileout.IsNull()) {
        fileout.fclose();
        remove(pathTmp);
        LogError("%s: Failed to open file %s\n", __func__, fs::PathToString(pathTmp));
        return false;
    }

    // Serialize
    if (!SerializeDB(fileout, data)) {
        fileout.fclose();
        remove(pathTmp);
        return false;
    }
    if (!fileout.Commit()) {
        fileout.fclose();
        remove(pathTmp);
        LogError("%s: Failed to flush file %s\n", __func__, fs::PathToString(pathTmp));
        return false;
    }
    fileout.fclose();

    // replace existing file, if any, with new file
    if (!RenameOver(pathTmp, path)) {
        remove(pathTmp);
        LogError("%s: Rename-into-place failed\n", __func__);
        return false;
    }

    return true;
}