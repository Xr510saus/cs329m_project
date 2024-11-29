// 8. Opens a file and reads its content
std::string read_file_content(const std::string& filename) {
    std::ifstream file(filename);
    std::string content;
    if (file.is_open()) {
        std::getline(file, content, '\0'); // Reads the entire file content
        file.close();
    }
    return content;
}