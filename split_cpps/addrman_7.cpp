void AddrManImpl::Serialize(Stream& s_) const
{
    LOCK(cs);

    /**
     * Serialized format.
     * * format version byte (@see `Format`)
     * * lowest compatible format version byte. This is used to help old software decide
     *   whether to parse the file. For example:
     *   * Bitcoin Core version N knows how to parse up to format=3. If a new format=4 is
     *     introduced in version N+1 that is compatible with format=3 and it is known that
     *     version N will be able to parse it, then version N+1 will write
     *     (format=4, lowest_compatible=3) in the first two bytes of the file, and so
     *     version N will still try to parse it.
     *   * Bitcoin Core version N+2 introduces a new incompatible format=5. It will write
     *     (format=5, lowest_compatible=5) and so any versions that do not know how to parse
     *     format=5 will not try to read the file.
     * * nKey
     * * nNew
     * * nTried
     * * number of "new" buckets XOR 2**30
     * * all new addresses (total count: nNew)
     * * all tried addresses (total count: nTried)
     * * for each new bucket:
     *   * number of elements
     *   * for each element: index in the serialized "all new addresses"
     * * asmap checksum
     *
     * 2**30 is xorred with the number of buckets to make addrman deserializer v0 detect it
     * as incompatible. This is necessary because it did not check the version number on
     * deserialization.
     *
     * vvNew, vvTried, mapInfo, mapAddr and vRandom are never encoded explicitly;
     * they are instead reconstructed from the other information.
     *
     * This format is more complex, but significantly smaller (at most 1.5 MiB), and supports
     * changes to the ADDRMAN_ parameters without breaking the on-disk structure.
     *
     * We don't use SERIALIZE_METHODS since the serialization and deserialization code has
     * very little in common.
     */

    // Always serialize in the latest version (FILE_FORMAT).
    ParamsStream s{s_, CAddress::V2_DISK};

    s << static_cast<uint8_t>(FILE_FORMAT);

    // Increment `lowest_compatible` iff a newly introduced format is incompatible with
    // the previous one.
    static constexpr uint8_t lowest_compatible = Format::V4_MULTIPORT;
    s << static_cast<uint8_t>(INCOMPATIBILITY_BASE + lowest_compatible);

    s << nKey;
    s << nNew;
    s << nTried;

    int nUBuckets = ADDRMAN_NEW_BUCKET_COUNT ^ (1 << 30);
    s << nUBuckets;
    std::unordered_map<nid_type, int> mapUnkIds;
    int nIds = 0;
    for (const auto& entry : mapInfo) {
        mapUnkIds[entry.first] = nIds;
        const AddrInfo& info = entry.second;
        if (info.nRefCount) {
            assert(nIds != nNew); // this means nNew was wrong, oh ow
            s << info;
            nIds++;
        }
    }
    nIds = 0;
    for (const auto& entry : mapInfo) {
        const AddrInfo& info = entry.second;
        if (info.fInTried) {
            assert(nIds != nTried); // this means nTried was wrong, oh ow
            s << info;
            nIds++;
        }
    }
    for (int bucket = 0; bucket < ADDRMAN_NEW_BUCKET_COUNT; bucket++) {
        int nSize = 0;
        for (int i = 0; i < ADDRMAN_BUCKET_SIZE; i++) {
            if (vvNew[bucket][i] != -1)
                nSize++;
        }
        s << nSize;
        for (int i = 0; i < ADDRMAN_BUCKET_SIZE; i++) {
            if (vvNew[bucket][i] != -1) {
                int nIndex = mapUnkIds[vvNew[bucket][i]];
                s << nIndex;
            }
        }
    }
    // Store asmap checksum after bucket entries so that it
    // can be ignored by older clients for backward compatibility.
    s << m_netgroupman.GetAsmapChecksum();
}