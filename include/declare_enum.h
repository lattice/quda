/*
 * A macro that declares an `enum class` as well as a `to_string` function for the enums.
 * The enum has also a default value `size` that measures the size of the enum.
 *
 * Credit: https://stackoverflow.com/a/71375077/12084612
 * -------
 * License: CC BY-SA 4.0
 * --------
 * Usage:
 * ------
 *
 * DECLARE_ENUM(WeekEnum, Mon, Tue, Wed, Thu, Fri, Sat, Sun,);
 *
 * int main()
 * {
 *     WeekEnum weekDay = WeekEnum::Wed;
 *     std::cout << to_string(weekDay) << std::endl; // prints Wed
 *     std::cout << to_string(WeekEnum::Sat) << std::endl; // prints Sat
 *     std::cout << to_string((int) WeekEnum::size) << std::endl; // prints 7
 *     return 0;
 * }
 *
 */

#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>

// Add the definition of this method into a cpp file. (only the declaration in the header)
static inline const std::vector<std::string> &get_enum_names(const std::string &en_key, const std::string &en_str)
{
  static std::unordered_map<std::string, std::vector<std::string>> en_names_map;
  const auto it = en_names_map.find(en_key);
  if (it != en_names_map.end()) return it->second;

  constexpr auto delim(',');
  std::vector<std::string> en_names;
  std::size_t start {};
  auto end = en_str.find(delim);
  while (end != std::string::npos) {
    while (en_str[start] == ' ') ++start;
    en_names.push_back(en_str.substr(start, end - start));
    start = end + 1;
    end = en_str.find(delim, start);
  }
  while (en_str[start] == ' ') ++start;
  en_names.push_back(en_str.substr(start));
  return en_names_map.emplace(en_key, std::move(en_names)).first->second;
}

#define DECLARE_ENUM(ENUM_NAME, ...)                                                                                   \
  enum class ENUM_NAME : unsigned int { __VA_ARGS__ size };                                                            \
  inline std::string to_string(ENUM_NAME en)                                                                           \
  {                                                                                                                    \
    const auto &names = get_enum_names(#ENUM_NAME #__VA_ARGS__, #__VA_ARGS__);                                         \
    return names[static_cast<std::size_t>(en)];                                                                        \
  }
