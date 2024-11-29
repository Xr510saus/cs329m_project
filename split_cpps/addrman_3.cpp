bool AddrInfo::IsTerrible(NodeSeconds now) const
{
    if (now - m_last_try <= 1min) { // never remove things tried in the last minute
        return false;
    }

    if (nTime > now + 10min) { // came in a flying DeLorean
        return true;
    }

    if (now - nTime > ADDRMAN_HORIZON) { // not seen in recent history
        return true;
    }

    if (TicksSinceEpoch<std::chrono::seconds>(m_last_success) == 0 && nAttempts >= ADDRMAN_RETRIES) { // tried N times and never a success
        return true;
    }

    if (now - m_last_success > ADDRMAN_MIN_FAIL && nAttempts >= ADDRMAN_MAX_FAILURES) { // N successive failures in the last week
        return true;
    }

    return false;
}