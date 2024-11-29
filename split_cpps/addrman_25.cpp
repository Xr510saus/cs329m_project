void AddrManImpl::ResolveCollisions_()
{
    AssertLockHeld(cs);

    for (std::set<nid_type>::iterator it = m_tried_collisions.begin(); it != m_tried_collisions.end();) {
        nid_type id_new = *it;

        bool erase_collision = false;

        // If id_new not found in mapInfo remove it from m_tried_collisions
        if (mapInfo.count(id_new) != 1) {
            erase_collision = true;
        } else {
            AddrInfo& info_new = mapInfo[id_new];

            // Which tried bucket to move the entry to.
            int tried_bucket = info_new.GetTriedBucket(nKey, m_netgroupman);
            int tried_bucket_pos = info_new.GetBucketPosition(nKey, false, tried_bucket);
            if (!info_new.IsValid()) { // id_new may no longer map to a valid address
                erase_collision = true;
            } else if (vvTried[tried_bucket][tried_bucket_pos] != -1) { // The position in the tried bucket is not empty

                // Get the to-be-evicted address that is being tested
                nid_type id_old = vvTried[tried_bucket][tried_bucket_pos];
                AddrInfo& info_old = mapInfo[id_old];

                const auto current_time{Now<NodeSeconds>()};

                // Has successfully connected in last X hours
                if (current_time - info_old.m_last_success < ADDRMAN_REPLACEMENT) {
                    erase_collision = true;
                } else if (current_time - info_old.m_last_try < ADDRMAN_REPLACEMENT) { // attempted to connect and failed in last X hours

                    // Give address at least 60 seconds to successfully connect
                    if (current_time - info_old.m_last_try > 60s) {
                        LogDebug(BCLog::ADDRMAN, "Replacing %s with %s in tried table\n", info_old.ToStringAddrPort(), info_new.ToStringAddrPort());

                        // Replaces an existing address already in the tried table with the new address
                        Good_(info_new, false, current_time);
                        erase_collision = true;
                    }
                } else if (current_time - info_new.m_last_success > ADDRMAN_TEST_WINDOW) {
                    // If the collision hasn't resolved in some reasonable amount of time,
                    // just evict the old entry -- we must not be able to
                    // connect to it for some reason.
                    LogDebug(BCLog::ADDRMAN, "Unable to test; replacing %s with %s in tried table anyway\n", info_old.ToStringAddrPort(), info_new.ToStringAddrPort());
                    Good_(info_new, false, current_time);
                    erase_collision = true;
                }
            } else { // Collision is not actually a collision anymore
                Good_(info_new, false, Now<NodeSeconds>());
                erase_collision = true;
            }
        }

        if (erase_collision) {
            m_tried_collisions.erase(it++);
        } else {
            it++;
        }
    }
}