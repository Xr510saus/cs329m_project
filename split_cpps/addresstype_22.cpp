bool IsValidDestination(const CTxDestination& dest) {
    return std::visit(ValidDestinationVisitor(), dest);
}