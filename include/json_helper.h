#pragma once

#include "externals/json.hpp"
using json = nlohmann::json;

void to_json(json &j, const dim3 &p) { j = json {{"x", p.x}, {"y", p.y}, {"z", p.z}}; }

void from_json(const json &j, dim3 &p)
{
  j.at("x").get_to(p.x);
  j.at("y").get_to(p.y);
  j.at("z").get_to(p.z);
}

void to_json(json &j, const int4 &p) { j = json {{"x", p.x}, {"y", p.y}, {"z", p.z}, {"w", p.w}}; }

void from_json(const json &j, int4 &p)
{
  j.at("x").get_to(p.x);
  j.at("y").get_to(p.y);
  j.at("z").get_to(p.z);
  j.at("w").get_to(p.w);
}
