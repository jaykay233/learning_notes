#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

inline bool ReadFile(const std::string &path, size_t expectBytes, void *dst,
                     size_t dstBytes) {
  if (dstBytes < expectBytes) {
    return false;
  }
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    std::printf("ERROR: cannot open %s\n", path.c_str());
    return false;
  }
  ifs.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(expectBytes));
  return static_cast<bool>(ifs) || ifs.gcount() == static_cast<std::streamsize>(expectBytes);
}

inline bool WriteFile(const std::string &path, const void *src, size_t bytes) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    std::printf("ERROR: cannot write %s\n", path.c_str());
    return false;
  }
  ofs.write(reinterpret_cast<const char *>(src), static_cast<std::streamsize>(bytes));
  return static_cast<bool>(ofs);
}

#endif
