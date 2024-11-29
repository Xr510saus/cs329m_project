int AddrManImpl::CheckAddrman() const
{
    AssertLockHeld(cs);

    LOG_TIME_MILLIS_WITH_CATEGORY_MSG_ONCE(
        strprintf("new %i, tried %i, total %u", nNew, nTried, vRandom.size()), BCLog::ADDRMAN);

    std::unordered_set<nid_type> setTried;
    std::unordered_map<nid_type, int> mapNew;
    std::unordered_map<Network, NewTriedCount> local_counts;

    if (vRandom.size() != (size_t)(nTried + nNew))
        return -7;

    for (const auto& entry : mapInfo) {
        nid_type n = entry.first;
        const AddrInfo& info = entry.second;
        if (info.fInTried) {
            if (!TicksSinceEpoch<std::chrono::seconds>(info.m_last_success)) {
                return -1;
            }
            if (info.nRefCount)
                return -2;
            setTried.insert(n);
            local_counts[info.GetNetwork()].n_tried++;
        } else {
            if (info.nRefCount < 0 || info.nRefCount > ADDRMAN_NEW_BUCKETS_PER_ADDRESS)
                return -3;
            if (!info.nRefCount)
                return -4;
            mapNew[n] = info.nRefCount;
            local_counts[info.GetNetwork()].n_new++;
        }
        const auto it{mapAddr.find(info)};
        if (it == mapAddr.end() || it->second != n) {
            return -5;
        }
        if (info.nRandomPos < 0 || (size_t)info.nRandomPos >= vRandom.size() || vRandom[info.nRandomPos] != n)
            return -14;
        if (info.m_last_try < NodeSeconds{0s}) {
            return -6;
        }
        if (info.m_last_success < NodeSeconds{0s}) {
            return -8;
        }
    }

    if (setTried.size() != (size_t)nTried)
        return -9;
    if (mapNew.size() != (size_t)nNew)
        return -10;

    for (int n = 0; n < ADDRMAN_TRIED_BUCKET_COUNT; n++) {
        for (int i = 0; i < ADDRMAN_BUCKET_SIZE; i++) {
            if (vvTried[n][i] != -1) {
                if (!setTried.count(vvTried[n][i]))
                    return -11;
                const auto it{mapInfo.find(vvTried[n][i])};
                if (it == mapInfo.end() || it->second.GetTriedBucket(nKey, m_netgroupman) != n) {
                    return -17;
                }
                if (it->second.GetBucketPosition(nKey, false, n) != i) {
                    return -18;
                }
                setTried.erase(vvTried[n][i]);
            }
        }
    }

    for (int n = 0; n < ADDRMAN_NEW_BUCKET_COUNT; n++) {
        for (int i = 0; i < ADDRMAN_BUCKET_SIZE; i++) {
            if (vvNew[n][i] != -1) {
                if (!mapNew.count(vvNew[n][i]))
                    return -12;
                const auto it{mapInfo.find(vvNew[n][i])};
                if (it == mapInfo.end() || it->second.GetBucketPosition(nKey, true, n) != i) {
                    return -19;
                }
                if (--mapNew[vvNew[n][i]] == 0)
                    mapNew.erase(vvNew[n][i]);
            }
        }
    }

    if (setTried.size())
        return -13;
    if (mapNew.size())
        return -15;
    if (nKey.IsNull())
        return -16;

    // It's possible that m_network_counts may have all-zero entries that local_counts
    // doesn't have if addrs from a network were being added and then removed again in the past.
    if (m_network_counts.size() < local_counts.size()) {
        return -20;
    }
    for (const auto& [net, count] : m_network_counts) {
        if (local_counts[net].n_new != count.n_new || local_counts[net].n_tried != count.n_tried) {
            return -21;
        }
    }

    return 0;
}