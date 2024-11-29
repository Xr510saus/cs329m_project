void DeserializeFileDB(const fs::path& path, Data&& data)
{
    FILE* file = fsbridge::fopen(path, "rb");
    AutoFile filein{file};
    if (filein.IsNull()) {
        throw DbNotFoundError{};
    }
    DeserializeDB(filein, data);
}