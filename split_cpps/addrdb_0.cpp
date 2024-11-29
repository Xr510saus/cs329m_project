bool SerializeDB(Stream& stream, const Data& data)
{
    // Write and commit header, data
    try {
        HashedSourceWriter hashwriter{stream};
        hashwriter << Params().MessageStart() << data;
        stream << hashwriter.GetHash();
    } catch (const std::exception& e) {
        LogError("%s: Serialize or I/O error - %s\n", __func__, e.what());
        return false;
    }

    return true;
}