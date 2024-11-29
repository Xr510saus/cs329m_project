double AddrInfo::GetChance(NodeSeconds now) const
{
    double fChance = 1.0;

    // deprioritize very recent attempts away
    if (now - m_last_try < 10min) {
        fChance *= 0.01;
    }

    // deprioritize 66% after each failed attempt, but at most 1/28th to avoid the search taking forever or overly penalizing outages.
    fChance *= pow(0.66, std::min(nAttempts, 8));

    return fChance;
}