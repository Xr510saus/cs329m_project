void AddrManImpl::MakeTried(AddrInfo& info, nid_type nId)
{
    AssertLockHeld(cs);

    // remove the entry from all new buckets
    const int start_bucket{info.GetNewBucket(nKey, m_netgroupman)};
    for (int n = 0; n < ADDRMAN_NEW_BUCKET_COUNT; ++n) {
        const int bucket{(start_bucket + n) % ADDRMAN_NEW_BUCKET_COUNT};
        const int pos{info.GetBucketPosition(nKey, true, bucket)};
        if (vvNew[bucket][pos] == nId) {
            vvNew[bucket][pos] = -1;
            info.nRefCount--;
            if (info.nRefCount == 0) break;
        }
    }
    nNew--;
    m_network_counts[info.GetNetwork()].n_new--;

    assert(info.nRefCount == 0);

    // which tried bucket to move the entry to
    int nKBucket = info.GetTriedBucket(nKey, m_netgroupman);
    int nKBucketPos = info.GetBucketPosition(nKey, false, nKBucket);

    // first make space to add it (the existing tried entry there is moved to new, deleting whatever is there).
    if (vvTried[nKBucket][nKBucketPos] != -1) {
        // find an item to evict
        nid_type nIdEvict = vvTried[nKBucket][nKBucketPos];
        assert(mapInfo.count(nIdEvict) == 1);
        AddrInfo& infoOld = mapInfo[nIdEvict];

        // Remove the to-be-evicted item from the tried set.
        infoOld.fInTried = false;
        vvTried[nKBucket][nKBucketPos] = -1;
        nTried--;
        m_network_counts[infoOld.GetNetwork()].n_tried--;

        // find which new bucket it belongs to
        int nUBucket = infoOld.GetNewBucket(nKey, m_netgroupman);
        int nUBucketPos = infoOld.GetBucketPosition(nKey, true, nUBucket);
        ClearNew(nUBucket, nUBucketPos);
        assert(vvNew[nUBucket][nUBucketPos] == -1);

        // Enter it into the new set again.
        infoOld.nRefCount = 1;
        vvNew[nUBucket][nUBucketPos] = nIdEvict;
        nNew++;
        m_network_counts[infoOld.GetNetwork()].n_new++;
        LogDebug(BCLog::ADDRMAN, "Moved %s from tried[%i][%i] to new[%i][%i] to make space\n",
                 infoOld.ToStringAddrPort(), nKBucket, nKBucketPos, nUBucket, nUBucketPos);
    }
    assert(vvTried[nKBucket][nKBucketPos] == -1);

    vvTried[nKBucket][nKBucketPos] = nId;
    nTried++;
    info.fInTried = true;
    m_network_counts[info.GetNetwork()].n_tried++;
}