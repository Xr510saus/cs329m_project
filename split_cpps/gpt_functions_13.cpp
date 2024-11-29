std::unordered_map<int, std::string> database;

void update_record(int id, const std::string& new_value) {
    database[id] = new_value;
}