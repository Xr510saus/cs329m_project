void AddrManImpl::Unserialize(Stream& s_)
{
    LOCK(cs);

    assert(vRandom.empty());

    Format format;
    s_ >> Using<CustomUintFormatter<1>>(format);

    const auto ser_params = (format >= Format::V3_BIP155 ? CAddress::V2_DISK : CAddress::V1_DISK);
    ParamsStream s{s_, ser_params};

    uint8_t compat;
    s >> compat;
    if (compat < INCOMPATIBILITY_BASE) {
        throw std::ios_base::failure(strprintf(
            "Corrupted addrman database: The compat value (%u) "
            "is lower than the expected minimum value %u.",
            compat, INCOMPATIBILITY_BASE));
    }
    const uint8_t lowest_compatible = compat - INCOMPATIBILITY_BASE;
    if (lowest_compatible > FILE_FORMAT) {
        throw InvalidAddrManVersionError(strprintf(
            "Unsupported format of addrman database: %u. It is compatible with formats >=%u, "
            "but the maximum supported by this version of %s is %u.",
            uint8_t{format}, lowest_compatible, CLIENT_NAME, uint8_t{FILE_FORMAT}));
    }

    s >> nKey;
    s >> nNew;
    s >> nTried;
    int nUBuckets = 0;
    s >> nUBuckets;
    if (format >= Format::V1_DETERMINISTIC) {
        nUBuckets ^= (1 << 30);
    }

    if (nNew > ADDRMAN_NEW_BUCKET_COUNT * ADDRMAN_BUCKET_SIZE || nNew < 0) {
        throw std::ios_base::failure(
                strprintf("Corrupt AddrMan serialization: nNew=%d, should be in [0, %d]",
                    nNew,
                    ADDRMAN_NEW_BUCKET_COUNT * ADDRMAN_BUCKET_SIZE));
    }

    if (nTried > ADDRMAN_TRIED_BUCKET_COUNT * ADDRMAN_BUCKET_SIZE || nTried < 0) {
        throw std::ios_base::failure(
                strprintf("Corrupt AddrMan serialization: nTried=%d, should be in [0, %d]",
                    nTried,
                    ADDRMAN_TRIED_BUCKET_COUNT * ADDRMAN_BUCKET_SIZE));
    }

    // Deserialize entries from the new table.
    for (int n = 0; n < nNew; n++) {
        AddrInfo& info = mapInfo[n];
        s >> info;
        mapAddr[info] = n;
        info.nRandomPos = vRandom.size();
        vRandom.push_back(n);
        m_network_counts[info.GetNetwork()].n_new++;
    }
    nIdCount = nNew;

    // Deserialize entries from the tried table.
    int nLost = 0;
    for (int n = 0; n < nTried; n++) {
        AddrInfo info;
        s >> info;
        int nKBucket = info.GetTriedBucket(nKey, m_netgroupman);
        int nKBucketPos = info.GetBucketPosition(nKey, false, nKBucket);
        if (info.IsValid()
                && vvTried[nKBucket][nKBucketPos] == -1) {
            info.nRandomPos = vRandom.size();
            info.fInTried = true;
            vRandom.push_back(nIdCount);
            mapInfo[nIdCount] = info;
            mapAddr[info] = nIdCount;
            vvTried[nKBucket][nKBucketPos] = nIdCount;
            nIdCount++;
            m_network_counts[info.GetNetwork()].n_tried++;
        } else {
            nLost++;
        }
    }
    nTried -= nLost;

    // Store positions in the new table buckets to apply later (if possible).
    // An entry may appear in up to ADDRMAN_NEW_BUCKETS_PER_ADDRESS buckets,
    // so we store all bucket-entry_index pairs to iterate through later.
    std::vector<std::pair<int, int>> bucket_entries;

    for (int bucket = 0; bucket < nUBuckets; ++bucket) {
        int num_entries{0};
        s >> num_entries;
        for (int n = 0; n < num_entries; ++n) {
            int entry_index{0};
            s >> entry_index;
            if (entry_index >= 0 && entry_index < nNew) {
                bucket_entries.emplace_back(bucket, entry_index);
            }
        }
    }

    // If the bucket count and asmap checksum haven't changed, then attempt
    // to restore the entries to the buckets/positions they were in before
    // serialization.
    uint256 supplied_asmap_checksum{m_netgroupman.GetAsmapChecksum()};
    uint256 serialized_asmap_checksum;
    if (format >= Format::V2_ASMAP) {
        s >> serialized_asmap_checksum;
    }
    const bool restore_bucketing{nUBuckets == ADDRMAN_NEW_BUCKET_COUNT &&
        serialized_asmap_checksum == supplied_asmap_checksum};

    if (!restore_bucketing) {
        LogDebug(BCLog::ADDRMAN, "Bucketing method was updated, re-bucketing addrman entries from disk\n");
    }

    for (auto bucket_entry : bucket_entries) {
        int bucket{bucket_entry.first};
        const int entry_index{bucket_entry.second};
        AddrInfo& info = mapInfo[entry_index];

        // Don't store the entry in the new bucket if it's not a valid address for our addrman
        if (!info.IsValid()) continue;

        // The entry shouldn't appear in more than
        // ADDRMAN_NEW_BUCKETS_PER_ADDRESS. If it has already, just skip
        // this bucket_entry.
        if (info.nRefCount >= ADDRMAN_NEW_BUCKETS_PER_ADDRESS) continue;

        int bucket_position = info.GetBucketPosition(nKey, true, bucket);
        if (restore_bucketing && vvNew[bucket][bucket_position] == -1) {
            // Bucketing has not changed, using existing bucket positions for the new table
            vvNew[bucket][bucket_position] = entry_index;
            ++info.nRefCount;
        } else {
            // In case the new table data cannot be used (bucket count wrong or new asmap),
            // try to give them a reference based on their primary source address.
            bucket = info.GetNewBucket(nKey, m_netgroupman);
            bucket_position = info.GetBucketPosition(nKey, true, bucket);
            if (vvNew[bucket][bucket_position] == -1) {
                vvNew[bucket][bucket_position] = entry_index;
                ++info.nRefCount;
            }
        }
    }

    // Prune new entries with refcount 0 (as a result of collisions or invalid address).
    int nLostUnk = 0;
    for (auto it = mapInfo.cbegin(); it != mapInfo.cend(); ) {
        if (it->second.fInTried == false && it->second.nRefCount == 0) {
            const auto itCopy = it++;
            Delete(itCopy->first);
            ++nLostUnk;
        } else {
            ++it;
        }
    }
    if (nLost + nLostUnk > 0) {
        LogDebug(BCLog::ADDRMAN, "addrman lost %i new and %i tried addresses due to collisions or invalid addresses\n", nLostUnk, nLost);
    }

    const int check_code{CheckAddrman()};
    if (check_code != 0) {
        throw std::ios_base::failure(strprintf(
            "Corrupt data. Consistency check failed with code %s",
            check_code));
    }
}