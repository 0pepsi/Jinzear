  ## Fate/Z Lua bytecode deobfuscator for MiWiFi firmware.

  - Converts Fate/Z obfuscated Lua 5.1 bytecode back to standard Lua 5.1 bytecode that can be processed by unluac, luadec, or luac -l.

  ## Fate/Z format differences from standard Lua 5.1:
    - Magic: \x1bFate/Z\x1b (8 bytes) instead of \x1bLua (4 bytes)
    - Header: 16 bytes total (8 magic + version + format + endian + sizeof fields +extra byte)
    - No LUAC_NUM test number after header
    - String type tag: 0x07 instead of 0x04
    - Boolean type tag: 0x04 instead of 0x01
    - Number type tag (double): 0x06 instead of 0x03
    - Nil type tag: 0x03 (alternate nil, no payload) in addition to 0x00
    - Integer type tag: 0x0c (extension, 4-byte LE int32)
    - String constants XOR-encrypted (key = last byte of encrypted string)
    - Source name and debug info name strings: only the size field is stored,
      no data bytes follow (names are stripped from the bytecode)
    - Function header field order differs from standard Lua 5.1:
      Standard: source, linedefined, lastlinedefined, nups, numparams, is_vararg maxstacksize
      Fate/Z:   nups, source, numparams, linedefined, is_vararg, lastlinedefined, maxstacksize
      (interleaves DumpChar and DumpInt calls differently)
    - Opcodes are permuted (42-slot table mapped to standard 38 opcodes)
    - LEN opcode uses C field (non-standard; cleared during conversion)
    - LOADNIL uses standard encoding (A=start, B=end, B >= A)
    - Three comparison opcodes (EQ, LT, LE) have duplicate encodings
    - nups field in function headers is zeroed (obfuscation)

  Author: 0xmadvise 
