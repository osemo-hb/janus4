--[[
Atomic Ring Buffer Update for Adaptive Threshold

This Lua script provides atomic updates to the threshold state in Redis.
It implements a ring buffer for velocity history that survives concurrent
updates from multiple API instances.

Arguments:
  KEYS[1] = Base key for threshold state (e.g., "threshold:{session_id}")
  ARGV[1] = New velocity value (float)
  ARGV[2] = New last_vector (JSON array)
  ARGV[3] = Window size (int) - max velocities to retain
  ARGV[4] = TTL in seconds (int)

Returns:
  Array of current velocities (as strings, needs float conversion)

Redis keys used:
  {base_key}:velocities - LIST of velocity values (ring buffer)
  {base_key}:last_vector - STRING containing JSON of last embedding vector
--]]

local base_key = KEYS[1]
local velocity = tonumber(ARGV[1])
local last_vector = ARGV[2]
local window_size = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

local velocities_key = base_key .. ':velocities'
local last_vector_key = base_key .. ':last_vector'

-- Append new velocity to the list (right side)
redis.call('RPUSH', velocities_key, velocity)

-- Trim to window size (keep only last N elements)
-- LTRIM keeps elements from start to end (inclusive)
-- To keep last N, we trim from -N to -1
redis.call('LTRIM', velocities_key, -window_size, -1)

-- Update last vector
redis.call('SET', last_vector_key, last_vector)

-- Set TTL on both keys to prevent stale data accumulation
redis.call('EXPIRE', velocities_key, ttl)
redis.call('EXPIRE', last_vector_key, ttl)

-- Return current velocities for threshold calculation
return redis.call('LRANGE', velocities_key, 0, -1)
