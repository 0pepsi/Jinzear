#!/usr/bin/env python3
"""
jinzear - Fate/Z Lua Bytecode Decompiler
=========================================

Full self-contained decompiler for Fate/Z obfuscated Lua 5.1 bytecode.
No external tools required — goes directly from .lua bytecode to readable
Lua source code.

Supports both Fate/Z (\x1bFate/Z\x1b) and standard Lua 5.1 (\x1bLua) bytecode.

Fate/Z format differences from standard Lua 5.1:
  - Magic: \x1bFate/Z\x1b (8 bytes) instead of \x1bLua (4 bytes)
  - Header: 16 bytes (8 magic + version + format + endian + sizeof fields + extra)
  - No LUAC_NUM test number after header
  - Type tags remapped: string=0x07, bool=0x04, number=0x06, nil_alt=0x03, int=0x0c
  - String constants XOR-encrypted (key = last byte)
  - Source/debug names stripped (size only, no data)
  - Function header field order permuted
  - Opcodes permuted (42-slot table → standard 38)
  - LEN opcode uses C field (cleared during conversion)
  - nups field zeroed (inferred from CLOSURE pseudo-ops)

Author: 0xmadvise
"""

import struct
import sys
import os
import io
import argparse
from collections import defaultdict
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional

FATEZ_MAGIC = b'\x1bFate/Z\x1b'
LUA51_MAGIC = b'\x1bLua'

# Fate/Z type tags
FATEZ_NIL = 0x00
FATEZ_NIL_ALT = 0x03
FATEZ_BOOL = 0x04
FATEZ_NUMBER = 0x06
FATEZ_STRING = 0x07
FATEZ_INT = 0x0c

# Standard Lua 5.1 type tags
STD_NIL = 0x00
STD_BOOL = 0x01
STD_NUMBER = 0x03
STD_STRING = 0x04


class Op(IntEnum):
    """Standard Lua 5.1 opcodes."""
    MOVE = 0
    LOADK = 1
    LOADBOOL = 2
    LOADNIL = 3
    GETUPVAL = 4
    GETGLOBAL = 5
    GETTABLE = 6
    SETGLOBAL = 7
    SETUPVAL = 8
    SETTABLE = 9
    NEWTABLE = 10
    SELF = 11
    ADD = 12
    SUB = 13
    MUL = 14
    DIV = 15
    MOD = 16
    POW = 17
    UNM = 18
    NOT = 19
    LEN = 20
    CONCAT = 21
    JMP = 22
    EQ = 23
    LT = 24
    LE = 25
    TEST = 26
    TESTSET = 27
    CALL = 28
    TAILCALL = 29
    RETURN = 30
    FORLOOP = 31
    FORPREP = 32
    TFORLOOP = 33
    SETLIST = 34
    CLOSE = 35
    CLOSURE = 36
    VARARG = 37


# Fate/Z opcode → Standard Lua 5.1 opcode
FZ_TO_STD = {
    1: 36, 2: 20, 3: 25, 4: 18, 5: 23, 6: 1, 7: 34, 8: 30,
    9: 26, 10: 33, 11: 32, 12: 13, 13: 37, 14: 15, 15: 11,
    16: 28, 17: 9, 18: 4, 19: 23, 20: 24, 21: 21, 22: 24,
    23: 25, 24: 2, 25: 16, 26: 31, 27: 6, 28: 10, 29: 35,
    30: 29, 31: 22, 32: 19, 33: 17, 34: 14, 35: 27, 36: 0,
    37: 12, 38: 5, 39: 8, 40: 7, 41: 3,
}

# Instruction format types
IABX_OPS = {Op.LOADK, Op.GETGLOBAL, Op.SETGLOBAL, Op.CLOSURE}
IASBX_OPS = {Op.JMP, Op.FORLOOP, Op.FORPREP}

# Arithmetic / comparison opcode names for expression reconstruction
BINOP_SYMBOLS = {
    Op.ADD: '+', Op.SUB: '-', Op.MUL: '*', Op.DIV: '/',
    Op.MOD: '%', Op.POW: '^',
}

COMPARE_SYMBOLS = {Op.EQ: '==', Op.LT: '<', Op.LE: '<='}

# FB2INT conversion for NEWTABLE/SETLIST
def fb2int(x):
    """Convert floating-point byte to integer (NEWTABLE array/hash size)."""
    if x < 8:
        return x
    return ((x & 7) + 8) << ((x >> 3) - 1)


class BytecodeReader:
    """Low-level binary reader with position tracking."""

    def __init__(self, data):
        self.data = data
        self.off = 0
        self.sizeof_size_t = 4

    def u8(self):
        val = self.data[self.off]
        self.off += 1
        return val

    def u32(self):
        val = struct.unpack_from('<I', self.data, self.off)[0]
        self.off += 4
        return val

    def i32(self):
        val = struct.unpack_from('<i', self.data, self.off)[0]
        self.off += 4
        return val

    def f64(self):
        val = struct.unpack_from('<d', self.data, self.off)[0]
        self.off += 8
        return val

    def raw(self, n):
        val = self.data[self.off:self.off + n]
        self.off += n
        return val

    def size_t(self):
        if self.sizeof_size_t == 8:
            val = struct.unpack_from('<Q', self.data, self.off)[0]
            self.off += 8
        else:
            val = self.u32()
        return val

    def remaining(self):
        return len(self.data) - self.off


@dataclass
class Instruction:
    """Decoded Lua 5.1 instruction."""
    op: int
    a: int = 0
    b: int = 0
    c: int = 0
    bx: int = 0
    sbx: int = 0
    raw: int = 0
    pc: int = 0

    @staticmethod
    def decode(raw, pc=0):
        op = raw & 0x3F
        a = (raw >> 6) & 0xFF
        b = (raw >> 23) & 0x1FF
        c = (raw >> 14) & 0x1FF
        bx = (raw >> 14) & 0x3FFFF
        sbx = bx - 131071
        return Instruction(op=op, a=a, b=b, c=c, bx=bx, sbx=sbx, raw=raw, pc=pc)


@dataclass
class Constant:
    """Lua constant value."""
    type: str  # 'nil', 'bool', 'number', 'string'
    value: object = None

    def to_lua(self):
        if self.type == 'nil':
            return 'nil'
        elif self.type == 'bool':
            return 'true' if self.value else 'false'
        elif self.type == 'number':
            v = self.value
            if isinstance(v, float):
                if v == int(v) and not (v == 0.0 and str(v) == '-0.0'):
                    iv = int(v)
                    if -2**31 <= iv <= 2**31 - 1:
                        return str(iv)
                return repr(v)
            return str(v)
        elif self.type == 'string':
            return lua_string_repr(self.value)
        return 'nil'


def lua_string_repr(s):
    """Convert a string to Lua string literal."""
    if isinstance(s, bytes):
        try:
            s = s.decode('utf-8')
        except UnicodeDecodeError:
            # Use byte escapes
            parts = []
            for b in s:
                if 32 <= b < 127 and b != ord('"') and b != ord('\\'):
                    parts.append(chr(b))
                elif b == ord('"'):
                    parts.append('\\"')
                elif b == ord('\\'):
                    parts.append('\\\\')
                elif b == ord('\n'):
                    parts.append('\\n')
                elif b == ord('\r'):
                    parts.append('\\r')
                elif b == ord('\t'):
                    parts.append('\\t')
                elif b == 0:
                    parts.append('\\0')
                else:
                    parts.append(f'\\{b}')
            return '"' + ''.join(parts) + '"'

    result = []
    for ch in s:
        if ch == '"':
            result.append('\\"')
        elif ch == '\\':
            result.append('\\\\')
        elif ch == '\n':
            result.append('\\n')
        elif ch == '\r':
            result.append('\\r')
        elif ch == '\t':
            result.append('\\t')
        elif ch == '\0':
            result.append('\\0')
        elif ord(ch) < 32 or ord(ch) > 126:
            result.append(f'\\{ord(ch)}')
        else:
            result.append(ch)
    return '"' + ''.join(result) + '"'



@dataclass
class Proto:
    """Lua function prototype (parsed bytecode)."""
    source: str = ''
    line_defined: int = 0
    last_line_defined: int = 0
    num_upvalues: int = 0
    num_params: int = 0
    is_vararg: int = 0
    max_stack_size: int = 0
    code: list = field(default_factory=list)  # list of Instruction
    constants: list = field(default_factory=list)  # list of Constant
    protos: list = field(default_factory=list)  # nested Proto list
    lineinfo: list = field(default_factory=list)
    locvars: list = field(default_factory=list)  # (name, startpc, endpc)
    upvalues: list = field(default_factory=list)  # upvalue names
    name: str = ''  # computed identifier


def decrypt_fatez_string(enc):
    """Decrypt XOR-encrypted Fate/Z string constant. Key = last byte."""
    if not enc:
        return enc
    key = enc[-1]
    return bytes([b ^ key for b in enc])


class Loader:
    """Loads Lua 5.1 or Fate/Z bytecode into Proto structures."""

    def __init__(self, data):
        self.reader = BytecodeReader(data)
        self.is_fatez = data[:8] == FATEZ_MAGIC
        self.is_lua51 = data[:4] == LUA51_MAGIC

        if not self.is_fatez and not self.is_lua51:
            raise ValueError(f"Unknown format: {data[:8].hex()}")

    def load(self):
        """Parse the header and top-level function."""
        r = self.reader

        if self.is_fatez:
            r.off = 8  # skip magic
            version = r.u8()
            fmt = r.u8()
            endian = r.u8()
            sizeof_int = r.u8()
            sizeof_size_t = r.u8()
            sizeof_instr = r.u8()
            sizeof_number = r.u8()
            extra = r.u8()  # Fate/Z extra byte
            r.sizeof_size_t = sizeof_size_t
        else:
            r.off = 4  # skip magic
            version = r.u8()
            fmt = r.u8()
            endian = r.u8()
            sizeof_int = r.u8()
            sizeof_size_t = r.u8()
            sizeof_instr = r.u8()
            sizeof_number = r.u8()
            integral = r.u8()
            r.sizeof_size_t = sizeof_size_t
            # Standard Lua 5.1 has LUAC_NUM test (8 bytes)
            _test_num = r.raw(sizeof_number)

        return self._load_function(0)

    def _load_function(self, depth, parent_nups=0):
        """Load one function prototype."""
        r = self.reader
        proto = Proto()

        if self.is_fatez:
            # Fate/Z field order: nups, source, numparams, linedef, is_vararg,
            #                     lastlinedef, maxstacksize
            nups_raw = r.u8()
            src_size = r.u32()  # size only, no data in Fate/Z
            proto.num_params = r.u8()
            proto.line_defined = r.u32()
            proto.is_vararg = r.u8()
            proto.last_line_defined = r.u32()
            proto.max_stack_size = r.u8()

            # Infer nups from parent CLOSURE pseudo-ops
            proto.num_upvalues = parent_nups if (depth > 0 and nups_raw == 0) else nups_raw
            proto.source = ''
        else:
            # Standard Lua 5.1 field order
            src_size = r.size_t()
            if src_size > 0:
                proto.source = r.raw(src_size).rstrip(b'\x00').decode('utf-8', errors='replace')
            proto.line_defined = r.u32()
            proto.last_line_defined = r.u32()
            proto.num_upvalues = r.u8()
            proto.num_params = r.u8()
            proto.is_vararg = r.u8()
            proto.max_stack_size = r.u8()

        # Instructions
        n_code = r.u32()
        raw_instructions = []
        for i in range(n_code):
            raw = r.u32()
            if self.is_fatez:
                raw = self._remap_fz_instruction(raw)
            inst = Instruction.decode(raw, pc=i)
            proto.code.append(inst)
            raw_instructions.append(raw)

        # Constants
        n_const = r.u32()
        for ci in range(n_const):
            t = r.u8()

            if self.is_fatez:
                base_t = t & 0x7F
                if base_t in (FATEZ_NIL, FATEZ_NIL_ALT):
                    proto.constants.append(Constant('nil'))
                elif base_t == FATEZ_BOOL:
                    proto.constants.append(Constant('bool', r.u8() != 0))
                elif base_t == FATEZ_NUMBER:
                    proto.constants.append(Constant('number', r.f64()))
                elif base_t == FATEZ_STRING:
                    ssize = r.u32()
                    if ssize > 0:
                        enc = r.raw(ssize)
                        dec = decrypt_fatez_string(enc)
                        # Strip trailing null
                        proto.constants.append(Constant('string', dec[:-1] if dec[-1:] == b'\x00' else dec))
                    else:
                        proto.constants.append(Constant('string', b''))
                elif base_t == FATEZ_INT:
                    proto.constants.append(Constant('number', float(r.i32())))
                else:
                    raise ValueError(f"Unknown Fate/Z const type 0x{t:02x} at offset 0x{r.off-1:x}")
            else:
                # Standard Lua 5.1 type tags
                if t == STD_NIL:
                    proto.constants.append(Constant('nil'))
                elif t == STD_BOOL:
                    proto.constants.append(Constant('bool', r.u8() != 0))
                elif t == STD_NUMBER:
                    proto.constants.append(Constant('number', r.f64()))
                elif t == STD_STRING:
                    ssize = r.size_t()
                    if ssize > 0:
                        s = r.raw(ssize)
                        proto.constants.append(Constant('string', s[:-1] if s[-1:] == b'\x00' else s))
                    else:
                        proto.constants.append(Constant('string', b''))
                else:
                    raise ValueError(f"Unknown const type 0x{t:02x}")

        # Infer nups for nested protos from CLOSURE pseudo-ops
        proto_nups_list = self._infer_child_nups(proto.code)

        # Nested protos
        n_protos = r.u32()
        for pi in range(n_protos):
            child_nups = proto_nups_list[pi] if pi < len(proto_nups_list) else 0
            child = self._load_function(depth + 1, parent_nups=child_nups)
            proto.protos.append(child)

        # Debug info: line info
        n_lineinfo = r.u32()
        for _ in range(n_lineinfo):
            proto.lineinfo.append(r.u32())

        # Debug info: local variables
        n_locvars = r.u32()
        for _ in range(n_locvars):
            if self.is_fatez:
                name_size = r.u32()  # size only, no data
                name = ''
            else:
                name_size = r.size_t()
                if name_size > 0:
                    name = r.raw(name_size).rstrip(b'\x00').decode('utf-8', errors='replace')
                else:
                    name = ''
            startpc = r.u32()
            endpc = r.u32()
            proto.locvars.append((name, startpc, endpc))

        # Debug info: upvalue names
        n_upvalues = r.u32()
        for _ in range(n_upvalues):
            if self.is_fatez:
                uv_size = r.u32()  # size only
                name = ''
            else:
                uv_size = r.size_t()
                if uv_size > 0:
                    name = r.raw(uv_size).rstrip(b'\x00').decode('utf-8', errors='replace')
                else:
                    name = ''
            proto.upvalues.append(name)

        return proto

    def _remap_fz_instruction(self, raw):
        """Remap Fate/Z opcode to standard Lua 5.1."""
        fz_op = raw & 0x3F
        if fz_op == 0 or fz_op not in FZ_TO_STD:
            return raw
        std_op = FZ_TO_STD[fz_op]
        new = (raw & ~0x3F) | std_op
        # LEN: clear non-standard C field
        if fz_op == 2:
            new = new & ~(0x1FF << 14)
        return new

    def _infer_child_nups(self, code):
        """Count upvalue pseudo-ops after each CLOSURE instruction."""
        result = []
        i = 0
        while i < len(code):
            if code[i].op == Op.CLOSURE:
                nups = 0
                j = i + 1
                while j < len(code):
                    if code[j].op in (Op.MOVE, Op.GETUPVAL):
                        nups += 1
                        j += 1
                    else:
                        break
                result.append(nups)
                i = j
            else:
                i += 1
        return result


_OP_NAMES = [
    'MOVE', 'LOADK', 'LOADBOOL', 'LOADNIL', 'GETUPVAL', 'GETGLOBAL',
    'GETTABLE', 'SETGLOBAL', 'SETUPVAL', 'SETTABLE', 'NEWTABLE', 'SELF',
    'ADD', 'SUB', 'MUL', 'DIV', 'MOD', 'POW', 'UNM', 'NOT', 'LEN',
    'CONCAT', 'JMP', 'EQ', 'LT', 'LE', 'TEST', 'TESTSET', 'CALL',
    'TAILCALL', 'RETURN', 'FORLOOP', 'FORPREP', 'TFORLOOP', 'SETLIST',
    'CLOSE', 'CLOSURE', 'VARARG',
]


def disassemble_proto(proto, out, depth=0, func_id='0'):
    """Disassemble a function prototype (luac -l style)."""
    prefix = '  ' * depth
    line = proto.line_defined
    last = proto.last_line_defined

    if depth == 0:
        source = proto.source or '?'
        out.write(f'\nmain <{source}:{line},{last}> '
                  f'({len(proto.code)} instructions)\n')
    else:
        source = proto.source or '?'
        out.write(f'\n{prefix}function <{source}:{line},{last}> '
                  f'({len(proto.code)} instructions)\n')

    out.write(f'{prefix}{proto.num_params}{"+" if proto.is_vararg else ""} params, '
              f'{proto.max_stack_size} slots, {proto.num_upvalues} upvalues, '
              f'{sum(1 for n, s, e in proto.locvars if n)} locals, '
              f'{len(proto.constants)} constants, {len(proto.protos)} functions\n')

    for inst in proto.code:
        lineno = proto.lineinfo[inst.pc] if inst.pc < len(proto.lineinfo) else 0
        op_name = _OP_NAMES[inst.op] if inst.op < len(_OP_NAMES) else f'OP_{inst.op}'

        if inst.op in IABX_OPS:
            args = f'{inst.a} {inst.bx}'
        elif inst.op in IASBX_OPS:
            args = f'{inst.a} {inst.sbx}'
        else:
            args = f'{inst.a} {inst.b} {inst.c}'

        comment = _get_inst_comment(inst, proto)
        line_str = f'[{lineno}]' if lineno else '[-]'

        out.write(f'{prefix}\t{inst.pc + 1}\t{line_str}\t{op_name:12s}\t{args}')
        if comment:
            out.write(f'\t; {comment}')
        out.write('\n')

    # Recurse into nested protos
    for i, child in enumerate(proto.protos):
        child_id = f'{func_id}_{i}'
        disassemble_proto(child, out, depth + 1, child_id)


def _get_inst_comment(inst, proto):
    """Generate comment for instruction (constant names, etc.)."""
    def kst(idx):
        if idx < len(proto.constants):
            return proto.constants[idx].to_lua()
        return f'K[{idx}]'

    def rk(val):
        if val >= 256:
            return kst(val - 256)
        return f'R{val}'

    op = inst.op
    if op in (Op.LOADK,):
        return kst(inst.bx)
    elif op in (Op.GETGLOBAL, Op.SETGLOBAL):
        return kst(inst.bx)
    elif op == Op.GETTABLE:
        return rk(inst.c)
    elif op == Op.SETTABLE:
        return f'{rk(inst.b)} {rk(inst.c)}'
    elif op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD, Op.POW):
        return f'{rk(inst.b)} {rk(inst.c)}'
    elif op == Op.SELF:
        return rk(inst.c)
    elif op in (Op.EQ, Op.LT, Op.LE):
        return f'{rk(inst.b)} {rk(inst.c)}'
    elif op == Op.CLOSURE:
        return f'proto {inst.bx}'
    elif op == Op.JMP:
        return f'-> {inst.pc + 1 + inst.sbx + 1}'
    elif op == Op.FORLOOP:
        return f'-> {inst.pc + 1 + inst.sbx + 1}'
    elif op == Op.FORPREP:
        return f'-> {inst.pc + 1 + inst.sbx + 1}'
    return ''


@dataclass
class BasicBlock:
    """A basic block in the control flow graph."""
    start: int  # first PC (inclusive)
    end: int  # last PC (inclusive)
    succs: list = field(default_factory=list)
    preds: list = field(default_factory=list)
    dom: Optional[int] = None  # immediate dominator block index


def build_cfg(code):
    """Build a control flow graph from instructions."""
    n = len(code)
    if n == 0:
        return [], {}

    # Find basic block leaders
    leaders = {0}
    for i, inst in enumerate(code):
        op = inst.op
        if op == Op.JMP:
            target = i + 1 + inst.sbx
            leaders.add(target)
            if i + 1 < n:
                leaders.add(i + 1)
        elif op in (Op.EQ, Op.LT, Op.LE, Op.TEST, Op.TESTSET):
            if i + 1 < n:
                leaders.add(i + 1)
            if i + 2 < n:
                leaders.add(i + 2)
        elif op in (Op.FORLOOP, Op.FORPREP):
            target = i + 1 + inst.sbx
            leaders.add(target)
            if i + 1 < n:
                leaders.add(i + 1)
        elif op == Op.TFORLOOP:
            if i + 1 < n:
                leaders.add(i + 1)
            if i + 2 < n:
                leaders.add(i + 2)
        elif op == Op.LOADBOOL and inst.c != 0:
            if i + 2 < n:
                leaders.add(i + 2)
            if i + 1 < n:
                leaders.add(i + 1)
        elif op == Op.RETURN:
            if i + 1 < n:
                leaders.add(i + 1)

    # Clamp leaders to valid range
    leaders = sorted(l for l in leaders if 0 <= l < n)

    # Build blocks
    blocks = []
    pc_to_block = {}
    for idx, leader in enumerate(leaders):
        end = leaders[idx + 1] - 1 if idx + 1 < len(leaders) else n - 1
        bb = BasicBlock(start=leader, end=end)
        blocks.append(bb)
        for pc in range(leader, end + 1):
            pc_to_block[pc] = idx

    # Connect edges
    for idx, bb in enumerate(blocks):
        last_inst = code[bb.end]
        op = last_inst.op

        if op == Op.JMP:
            target = bb.end + 1 + last_inst.sbx
            if target in pc_to_block:
                bb.succs.append(pc_to_block[target])
        elif op in (Op.EQ, Op.LT, Op.LE, Op.TEST, Op.TESTSET):
            # Fall through and skip
            if bb.end + 1 in pc_to_block:
                bb.succs.append(pc_to_block[bb.end + 1])
            if bb.end + 2 in pc_to_block:
                bb.succs.append(pc_to_block[bb.end + 2])
        elif op in (Op.FORLOOP,):
            target = bb.end + 1 + last_inst.sbx
            if target in pc_to_block:
                bb.succs.append(pc_to_block[target])
            if bb.end + 1 in pc_to_block:
                bb.succs.append(pc_to_block[bb.end + 1])
        elif op in (Op.FORPREP,):
            target = bb.end + 1 + last_inst.sbx
            if target in pc_to_block:
                bb.succs.append(pc_to_block[target])
        elif op == Op.TFORLOOP:
            if bb.end + 1 in pc_to_block:
                bb.succs.append(pc_to_block[bb.end + 1])
            if bb.end + 2 in pc_to_block:
                bb.succs.append(pc_to_block[bb.end + 2])
        elif op == Op.RETURN:
            pass  # no successors
        elif op == Op.LOADBOOL and last_inst.c != 0:
            if bb.end + 2 in pc_to_block:
                bb.succs.append(pc_to_block[bb.end + 2])
        else:
            # Fall through
            if idx + 1 < len(blocks):
                bb.succs.append(idx + 1)

        # Unique successors
        bb.succs = list(dict.fromkeys(bb.succs))

    # Build predecessor lists
    for idx, bb in enumerate(blocks):
        for s in bb.succs:
            blocks[s].preds.append(idx)

    return blocks, pc_to_block


import threading
_render_depth = threading.local()


def _safe_to_lua(expr):
    """Call to_lua with recursion depth protection."""
    depth = getattr(_render_depth, 'val', 0)
    if depth > 20:
        return '...'
    _render_depth.val = depth + 1
    try:
        return expr.to_lua()
    finally:
        _render_depth.val = depth


class Expr:
    """Base class for expression AST nodes."""
    precedence = 100

    def to_lua(self):
        return '??'

    def wrap(self, child):
        if child.precedence < self.precedence:
            return f'({_safe_to_lua(child)})'
        return _safe_to_lua(child)


class NilExpr(Expr):
    def to_lua(self):
        return 'nil'

class BoolExpr(Expr):
    def __init__(self, val):
        self.val = val
    def to_lua(self):
        return 'true' if self.val else 'false'

class NumberExpr(Expr):
    def __init__(self, val):
        self.val = val
    def to_lua(self):
        v = self.val
        if isinstance(v, float):
            if v == int(v) and not (v == 0.0 and str(v).startswith('-')):
                iv = int(v)
                if -2**53 <= iv <= 2**53:
                    return str(iv)
            return repr(v)
        return str(v)

class StringExpr(Expr):
    def __init__(self, val):
        self.val = val
    def to_lua(self):
        return lua_string_repr(self.val)

class VarExpr(Expr):
    """A register or named variable."""
    def __init__(self, name):
        self.name = name
    def to_lua(self):
        return self.name

class UpvalExpr(Expr):
    def __init__(self, name):
        self.name = name
    def to_lua(self):
        return self.name

class GlobalExpr(Expr):
    def __init__(self, name):
        self.name = name
    def to_lua(self):
        return self.name

class VarargExpr(Expr):
    def to_lua(self):
        return '...'

class TableConstructor(Expr):
    """Table constructor: { ... }"""
    def __init__(self):
        self.array_part = []  # list of Expr
        self.hash_part = []   # list of (key_expr, val_expr)
    def to_lua(self):
        parts = []
        for v in self.array_part:
            parts.append(_safe_to_lua(v))
        for k, v in self.hash_part:
            if isinstance(k, StringExpr):
                s = k.val
                if isinstance(s, bytes):
                    try:
                        s = s.decode('utf-8')
                    except UnicodeDecodeError:
                        parts.append(f'[{k.to_lua()}] = {_safe_to_lua(v)}')
                        continue
                if s.isidentifier():
                    parts.append(f'{s} = {_safe_to_lua(v)}')
                else:
                    parts.append(f'[{k.to_lua()}] = {_safe_to_lua(v)}')
            else:
                parts.append(f'[{k.to_lua()}] = {_safe_to_lua(v)}')
        if not parts:
            return '{}'
        if len(parts) <= 4:
            return '{' + ', '.join(parts) + '}'
        inner = ',\n  '.join(parts)
        return '{\n  ' + inner + '\n}'

class IndexExpr(Expr):
    """Table indexing: t[k] or t.k"""
    precedence = 95

    def __init__(self, table, key):
        self.table = table
        self.key = key

    def to_lua(self):
        t = _safe_to_lua(self.table)
        if isinstance(self.key, StringExpr):
            s = self.key.val
            if isinstance(s, bytes):
                try:
                    s = s.decode('utf-8')
                except UnicodeDecodeError:
                    return f'{t}[{_safe_to_lua(self.key)}]'
            if isinstance(s, str) and s.isidentifier():
                return f'{t}.{s}'
        return f'{t}[{_safe_to_lua(self.key)}]'

class MethodCallExpr(Expr):
    """Method call: obj:method(args)"""
    precedence = 95

    def __init__(self, obj, method, args):
        self.obj = obj
        self.method = method
        self.args = args

    def to_lua(self):
        o = _safe_to_lua(self.obj)
        m = self.method
        if isinstance(m, bytes):
            try:
                m = m.decode('utf-8')
            except UnicodeDecodeError:
                m = lua_string_repr(m)
        a = ', '.join(_safe_to_lua(x) for x in self.args)
        return f'{o}:{m}({a})'

class BinopExpr(Expr):
    """Binary operation."""
    _PREC = {
        'or': 10, 'and': 20,
        '<': 30, '>': 30, '<=': 30, '>=': 30, '~=': 30, '==': 30,
        '..': 40,
        '+': 50, '-': 50,
        '*': 60, '/': 60, '%': 60,
        '^': 80,
    }

    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right
        self.precedence = self._PREC.get(op, 50)

    def to_lua(self):
        l = _safe_to_lua(self.left)
        r = _safe_to_lua(self.right)
        if self.left.precedence < self.precedence:
            l = f'({l})'
        if self.right.precedence < self.precedence or (
                self.right.precedence == self.precedence and self.op not in ('..', '^')):
            r = f'({r})'
        return f'{l} {self.op} {r}'

class UnopExpr(Expr):
    """Unary operation."""
    precedence = 70

    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def to_lua(self):
        o = _safe_to_lua(self.operand)
        if self.operand.precedence < self.precedence:
            o = f'({o})'
        if self.op == 'not':
            return f'not {o}'
        return f'{self.op}{o}'

class ConcatExpr(Expr):
    """String concatenation: a .. b .. c"""
    precedence = 40

    def __init__(self, parts):
        self.parts = parts

    def to_lua(self):
        return ' .. '.join(_safe_to_lua(p) for p in self.parts)

class FuncCallExpr(Expr):
    """Function call: f(args)"""
    precedence = 95

    def __init__(self, func, args, is_method=False):
        self.func = func
        self.args = args
        self.is_method = is_method

    def to_lua(self):
        f = _safe_to_lua(self.func)
        a = ', '.join(_safe_to_lua(x) for x in self.args)
        return f'{f}({a})'

class ClosureExpr(Expr):
    """Function literal (closure)."""
    def __init__(self, proto_idx, source_text=''):
        self.proto_idx = proto_idx
        self.source_text = source_text

    def to_lua(self):
        return self.source_text or f'<closure:{self.proto_idx}>'


class Stmt:
    """Base class for statement AST nodes."""
    def to_lua(self, indent=0):
        return '  ' * indent + '-- ???'


class AssignStmt(Stmt):
    def __init__(self, targets, values):
        self.targets = targets  # list of Expr
        self.values = values    # list of Expr

    def to_lua(self, indent=0):
        pfx = '  ' * indent
        t = ', '.join(_safe_to_lua(x) for x in self.targets)
        v = ', '.join(_safe_to_lua(x) for x in self.values)
        return f'{pfx}{t} = {v}'


class LocalStmt(Stmt):
    def __init__(self, names, values=None):
        self.names = names  # list of str
        self.values = values or []  # list of Expr

    def to_lua(self, indent=0):
        pfx = '  ' * indent
        n = ', '.join(self.names)
        if self.values:
            v = ', '.join(_safe_to_lua(x) for x in self.values)
            return f'{pfx}local {n} = {v}'
        return f'{pfx}local {n}'


class ReturnStmt(Stmt):
    def __init__(self, values):
        self.values = values

    def to_lua(self, indent=0):
        pfx = '  ' * indent
        if not self.values:
            return f'{pfx}return'
        v = ', '.join(_safe_to_lua(x) for x in self.values)
        return f'{pfx}return {v}'


class CallStmt(Stmt):
    def __init__(self, expr):
        self.expr = expr

    def to_lua(self, indent=0):
        return '  ' * indent + _safe_to_lua(self.expr)


class IfStmt(Stmt):
    def __init__(self):
        self.branches = []  # list of (condition_expr_or_None, body_stmts)

    def to_lua(self, indent=0):
        pfx = '  ' * indent
        lines = []
        for i, (cond, body) in enumerate(self.branches):
            if i == 0:
                lines.append(f'{pfx}if {cond.to_lua()} then')
            elif cond is not None:
                lines.append(f'{pfx}elseif {cond.to_lua()} then')
            else:
                lines.append(f'{pfx}else')
            for s in body:
                lines.append(s.to_lua(indent + 1))
        lines.append(f'{pfx}end')
        return '\n'.join(lines)


class WhileStmt(Stmt):
    def __init__(self, cond, body):
        self.cond = cond
        self.body = body

    def to_lua(self, indent=0):
        pfx = '  ' * indent
        lines = [f'{pfx}while {_safe_to_lua(self.cond)} do']
        for s in self.body:
            lines.append(s.to_lua(indent + 1))
        lines.append(f'{pfx}end')
        return '\n'.join(lines)


class RepeatStmt(Stmt):
    def __init__(self, cond, body):
        self.cond = cond
        self.body = body

    def to_lua(self, indent=0):
        pfx = '  ' * indent
        lines = [f'{pfx}repeat']
        for s in self.body:
            lines.append(s.to_lua(indent + 1))
        lines.append(f'{pfx}until {_safe_to_lua(self.cond)}')
        return '\n'.join(lines)


class ForNumStmt(Stmt):
    def __init__(self, var, init, limit, step, body):
        self.var = var
        self.init = init
        self.limit = limit
        self.step = step
        self.body = body

    def to_lua(self, indent=0):
        pfx = '  ' * indent
        step_str = ''
        if isinstance(self.step, NumberExpr) and self.step.val == 1.0:
            step_str = ''
        else:
            step_str = f', {_safe_to_lua(self.step)}'
        lines = [f'{pfx}for {self.var} = {_safe_to_lua(self.init)}, {_safe_to_lua(self.limit)}{step_str} do']
        for s in self.body:
            lines.append(s.to_lua(indent + 1))
        lines.append(f'{pfx}end')
        return '\n'.join(lines)


class ForInStmt(Stmt):
    def __init__(self, vars, iterators, body):
        self.vars = vars
        self.iterators = iterators
        self.body = body

    def to_lua(self, indent=0):
        pfx = '  ' * indent
        v = ', '.join(self.vars)
        it = ', '.join(_safe_to_lua(x) for x in self.iterators)
        lines = [f'{pfx}for {v} in {it} do']
        for s in self.body:
            lines.append(s.to_lua(indent + 1))
        lines.append(f'{pfx}end')
        return '\n'.join(lines)


class DoStmt(Stmt):
    def __init__(self, body):
        self.body = body

    def to_lua(self, indent=0):
        pfx = '  ' * indent
        lines = [f'{pfx}do']
        for s in self.body:
            lines.append(s.to_lua(indent + 1))
        lines.append(f'{pfx}end')
        return '\n'.join(lines)


class BreakStmt(Stmt):
    def to_lua(self, indent=0):
        return '  ' * indent + 'break'


class CommentStmt(Stmt):
    def __init__(self, text):
        self.text = text

    def to_lua(self, indent=0):
        return '  ' * indent + '-- ' + self.text


class Decompiler:
    """Main decompilation engine: Proto -> Stmt list -> Lua source.

    Key design: register-tracking with lazy emission. Operations that only
    compute values (GETGLOBAL, LOADK, GETTABLE, arithmetic, etc.) update
    the register map but do NOT emit statements. Statements are emitted only
    when side effects occur: CALL (as statement), SETGLOBAL, SETUPVAL,
    SETTABLE (outside table constructor), RETURN, and control flow.

    This produces clean output like:
        module("foo", package.seeall)
    instead of:
        local l0 = module
        local l1 = "foo"
        local l2 = package
        l2 = l2.seeall
        l0(l1, l2)
    """

    def __init__(self, proto, parent=None, proto_idx=0):
        self.proto = proto
        self.parent = parent
        self.proto_idx = proto_idx
        self.code = proto.code
        self.n = len(self.code)
        self.constants = proto.constants

        # Register state: maps register -> Expr (lazy values)
        self.regs = {}
        # Which registers are "pending" (not yet emitted as local/assignment)
        self.pending = set()
        # Variable names
        self.locals = {}  # reg -> name
        self.local_declared = set()  # registers declared as local
        self.upval_names = list(proto.upvalues) if proto.upvalues else []
        self.pending_tables = {}  # reg -> TableConstructor
        self.local_counter = 0  # for generating unique local names

        self._init_locals()

    def _init_locals(self):
        """Initialize local variable names from debug info."""
        proto = self.proto

        # Build per-PC local variable maps from debug info
        self.locvar_by_reg = {}
        reg_idx = 0
        for lv_name, startpc, endpc in proto.locvars:
            self.locvar_by_reg[reg_idx] = (lv_name or '', startpc, endpc)
            reg_idx += 1

        # Parameters
        for i in range(proto.num_params):
            info = self.locvar_by_reg.get(i)
            name = info[0] if info and info[0] else f'a{i}'
            self.locals[i] = name
            self.local_declared.add(i)
            self.regs[i] = VarExpr(name)

        # Pre-analyze: count how many times each register is READ
        # If read more than once, it must be materialized into a local
        self.multi_read_regs = set()
        read_counts = defaultdict(int)
        for inst in self.code:
            op = inst.op
            reads = []
            if op == Op.MOVE:
                reads.append(inst.b)
            elif op == Op.SETGLOBAL or op == Op.SETUPVAL:
                reads.append(inst.a)
            elif op == Op.GETTABLE:
                reads.append(inst.b)
            elif op == Op.SETTABLE:
                reads.extend([inst.a])
            elif op == Op.SELF:
                reads.append(inst.b)
            elif op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD, Op.POW,
                        Op.EQ, Op.LT, Op.LE):
                if inst.b < 256: reads.append(inst.b)
                if inst.c < 256: reads.append(inst.c)
            elif op in (Op.UNM, Op.NOT, Op.LEN):
                reads.append(inst.b)
            elif op == Op.CONCAT:
                reads.extend(range(inst.b, inst.c + 1))
            elif op == Op.CALL or op == Op.TAILCALL:
                reads.append(inst.a)
                nargs = inst.b - 1 if inst.b != 0 else 0
                for j in range(max(nargs, 0)):
                    reads.append(inst.a + 1 + j)
            elif op == Op.RETURN:
                nret = inst.b - 1 if inst.b != 0 else 0
                for j in range(max(nret, 0)):
                    reads.append(inst.a + j)
            elif op == Op.TEST:
                reads.append(inst.a)
            elif op == Op.TESTSET:
                reads.append(inst.b)

            for r in reads:
                read_counts[r] += 1
                if read_counts[r] > 1:
                    self.multi_read_regs.add(r)

    def _reg_name(self, reg):
        """Get or generate a name for a register."""
        if reg in self.locals:
            return self.locals[reg]
        # Check debug info for a name at this register
        info = self.locvar_by_reg.get(reg)
        if info and info[0]:
            return info[0]
        # Generate one
        self.local_counter += 1
        return f'l_{self.local_counter}'

    def _get_reg(self, reg):
        """Get the expression in a register."""
        if reg in self.regs:
            return self.regs[reg]
        return VarExpr(self._reg_name(reg))

    def _set_reg(self, reg, expr):
        """Set a register value (lazy, no statement emitted)."""
        self.regs[reg] = expr
        self.pending.add(reg)
        # Auto-materialize into local stmts list (populated by _decompile_range)
        # We can't emit stmts here, but we mark for materialization

    def _maybe_materialize(self, reg, stmts):
        """If a register holds a complex expression and is read multiple times,
        emit it as a local to avoid duplicating side effects and deep nesting."""
        if reg not in self.regs:
            return
        expr = self.regs[reg]
        # Only materialize complex expressions (calls, table accesses, etc.)
        if isinstance(expr, (VarExpr, GlobalExpr, UpvalExpr, NilExpr,
                             BoolExpr, NumberExpr, StringExpr, VarargExpr,
                             ClosureExpr)):
            return  # simple or already handled, OK to inline
        if reg in self.multi_read_regs and reg not in self.local_declared:
            name = self._reg_name(reg)
            self.locals[reg] = name
            self.local_declared.add(reg)
            stmts.append(LocalStmt([name], [expr]))
            self.regs[reg] = VarExpr(name)
            self.pending.discard(reg)

    def _materialize_all_complex(self, stmts):
        """Force-materialize all registers with complex expressions to locals.
        Called before control flow to prevent expression trees from growing unbounded."""
        for reg in list(self.pending):
            if reg in self.local_declared:
                continue
            expr = self.regs.get(reg)
            if expr is None:
                continue
            if isinstance(expr, (VarExpr, GlobalExpr, UpvalExpr, NilExpr,
                                 BoolExpr, NumberExpr, StringExpr, VarargExpr,
                                 ClosureExpr)):
                continue
            name = self._reg_name(reg)
            self.locals[reg] = name
            self.local_declared.add(reg)
            stmts.append(LocalStmt([name], [expr]))
            self.regs[reg] = VarExpr(name)
        self.pending.clear()

    def _kst(self, idx):
        """Get constant as expression."""
        if idx < len(self.constants):
            c = self.constants[idx]
            if c.type == 'nil':
                return NilExpr()
            elif c.type == 'bool':
                return BoolExpr(c.value)
            elif c.type == 'number':
                return NumberExpr(c.value)
            elif c.type == 'string':
                return StringExpr(c.value)
        return VarExpr(f'K{idx}')

    def _rk(self, val):
        """Resolve RK operand (register or constant)."""
        if val >= 256:
            return self._kst(val - 256)
        return self._get_reg(val)

    def _upval_name(self, idx):
        if idx < len(self.upval_names) and self.upval_names[idx]:
            return self.upval_names[idx]
        return f'_upval{idx}'

    def _global_name(self, bx):
        """Extract global name from constant index."""
        c = self._kst(bx)
        if isinstance(c, StringExpr):
            name = c.val
            if isinstance(name, bytes):
                name = name.decode('utf-8', errors='replace')
            return name
        return c.to_lua()

    def _materialize_reg(self, reg, stmts):
        """Force a pending register to be emitted as a local statement.
        Used when a register needs to be read multiple times or across branches."""
        if reg in self.pending and reg not in self.local_declared:
            name = self._reg_name(reg)
            self.locals[reg] = name
            self.local_declared.add(reg)
            expr = self.regs.get(reg, NilExpr())
            stmts.append(LocalStmt([name], [expr]))
            self.regs[reg] = VarExpr(name)
            self.pending.discard(reg)

    def _is_reg_used_later(self, reg, pc, end):
        """Check if register is read by any later instruction before being overwritten.
        Scans at most 20 instructions ahead for performance."""
        limit = min(pc + 21, end, self.n)
        for i in range(pc + 1, limit):
            inst = self.code[i]
            # Check if this instruction READS the register
            reads = set()
            op = inst.op
            if op == Op.MOVE:
                reads.add(inst.b)
            elif op in (Op.GETGLOBAL, Op.SETGLOBAL):
                if op == Op.SETGLOBAL:
                    reads.add(inst.a)
            elif op == Op.GETTABLE:
                reads.add(inst.b)
                if inst.c < 256:
                    reads.add(inst.c)
            elif op == Op.SETTABLE:
                reads.add(inst.a)
                if inst.b < 256:
                    reads.add(inst.b)
                if inst.c < 256:
                    reads.add(inst.c)
            elif op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD, Op.POW):
                if inst.b < 256:
                    reads.add(inst.b)
                if inst.c < 256:
                    reads.add(inst.c)
            elif op in (Op.UNM, Op.NOT, Op.LEN):
                reads.add(inst.b)
            elif op == Op.CONCAT:
                for r in range(inst.b, inst.c + 1):
                    reads.add(r)
            elif op == Op.CALL or op == Op.TAILCALL:
                reads.add(inst.a)
                nargs = inst.b - 1 if inst.b != 0 else 0
                for j in range(max(nargs, 0)):
                    reads.add(inst.a + 1 + j)
            elif op == Op.RETURN:
                nret = inst.b - 1 if inst.b != 0 else 0
                for j in range(max(nret, 0)):
                    reads.add(inst.a + j)
            elif op == Op.SELF:
                reads.add(inst.b)
                if inst.c < 256:
                    reads.add(inst.c)
            elif op in (Op.EQ, Op.LT, Op.LE):
                if inst.b < 256:
                    reads.add(inst.b)
                if inst.c < 256:
                    reads.add(inst.c)
            elif op == Op.TEST:
                reads.add(inst.a)
            elif op == Op.TESTSET:
                reads.add(inst.b)
            elif op == Op.SETUPVAL:
                reads.add(inst.a)
            elif op == Op.SETLIST:
                n = inst.b if inst.b != 0 else 0
                for j in range(1, n + 1):
                    reads.add(inst.a + j)
            elif op == Op.FORPREP:
                reads.update({inst.a, inst.a + 1, inst.a + 2})

            if reg in reads:
                return True

            # Check if this instruction WRITES the register (kills it)
            writes = set()
            if op in (Op.MOVE, Op.LOADK, Op.LOADBOOL, Op.GETUPVAL, Op.GETGLOBAL,
                      Op.GETTABLE, Op.NEWTABLE, Op.SELF, Op.ADD, Op.SUB, Op.MUL,
                      Op.DIV, Op.MOD, Op.POW, Op.UNM, Op.NOT, Op.LEN, Op.CONCAT,
                      Op.CLOSURE, Op.VARARG):
                writes.add(inst.a)
            elif op == Op.LOADNIL:
                for r in range(inst.a, inst.b + 1):
                    writes.add(r)
            elif op == Op.CALL:
                nret = inst.c - 1 if inst.c != 0 else 1
                for r in range(max(nret, 1)):
                    writes.add(inst.a + r)

            if reg in writes:
                return False  # overwritten before read

        return False

    def decompile(self):
        """Decompile the function prototype to a list of statements."""
        self._branch_depth = 0
        self._visited_ranges = set()
        return self._decompile_range(0, self.n)

    def _decompile_range(self, start, end):
        """Decompile instructions [start, end) to statements.

        The core approach: track register values lazily and only emit
        statements when a side effect forces it.
        """
        range_key = (start, end)
        if range_key in getattr(self, '_visited_ranges', set()):
            return [CommentStmt(f'(cycle detected at PC {start}-{end})')]
        if hasattr(self, '_visited_ranges'):
            self._visited_ranges.add(range_key)

        stmts = []
        pc = start

        while pc < end and pc < self.n:
            inst = self.code[pc]
            op = inst.op

            # ===== Pure register operations (no statement emitted) =====

            if op == Op.MOVE:
                self._set_reg(inst.a, self._get_reg(inst.b))
                pc += 1

            elif op == Op.LOADK:
                self._set_reg(inst.a, self._kst(inst.bx))
                pc += 1

            elif op == Op.LOADBOOL:
                self._set_reg(inst.a, BoolExpr(inst.b != 0))
                if inst.c != 0:
                    pc += 2
                else:
                    pc += 1

            elif op == Op.LOADNIL:
                for r in range(inst.a, inst.b + 1):
                    self._set_reg(r, NilExpr())
                pc += 1

            elif op == Op.GETUPVAL:
                self._set_reg(inst.a, UpvalExpr(self._upval_name(inst.b)))
                pc += 1

            elif op == Op.GETGLOBAL:
                self._set_reg(inst.a, GlobalExpr(self._global_name(inst.bx)))
                pc += 1

            elif op == Op.GETTABLE:
                self._set_reg(inst.a, IndexExpr(self._get_reg(inst.b), self._rk(inst.c)))
                self._maybe_materialize(inst.a, stmts)
                pc += 1

            elif op == Op.NEWTABLE:
                tc = TableConstructor()
                self._set_reg(inst.a, tc)
                self.pending_tables[inst.a] = tc
                pc += 1

            elif op == Op.SELF:
                table = self._get_reg(inst.b)
                method_key = self._rk(inst.c)
                self._set_reg(inst.a + 1, table)
                self._set_reg(inst.a, IndexExpr(table, method_key))
                pc += 1

            elif op in BINOP_SYMBOLS:
                self._set_reg(inst.a, BinopExpr(BINOP_SYMBOLS[op],
                              self._rk(inst.b), self._rk(inst.c)))
                pc += 1

            elif op == Op.UNM:
                self._set_reg(inst.a, UnopExpr('-', self._get_reg(inst.b)))
                pc += 1

            elif op == Op.NOT:
                self._set_reg(inst.a, UnopExpr('not', self._get_reg(inst.b)))
                pc += 1

            elif op == Op.LEN:
                self._set_reg(inst.a, UnopExpr('#', self._get_reg(inst.b)))
                pc += 1

            elif op == Op.CONCAT:
                parts = [self._get_reg(r) for r in range(inst.b, inst.c + 1)]
                self._set_reg(inst.a, ConcatExpr(parts))
                pc += 1

            elif op == Op.CLOSURE:
                pidx = inst.bx
                if pidx < len(self.proto.protos):
                    child_proto = self.proto.protos[pidx]
                    try:
                        child_dc = Decompiler(child_proto, parent=self, proto_idx=pidx)
                        child_stmts = child_dc.decompile()
                        child_src = format_function(child_proto, child_stmts, inline=True)
                    except RecursionError:
                        child_src = 'function() --[[ decompiler: recursion limit ]] end'
                    self._set_reg(inst.a, ClosureExpr(pidx, child_src))
                    nups = child_proto.num_upvalues
                else:
                    self._set_reg(inst.a, ClosureExpr(pidx))
                    nups = 0
                pc += 1 + nups

            elif op == Op.VARARG:
                self._set_reg(inst.a, VarargExpr())
                n = inst.b - 1 if inst.b != 0 else -1
                if n > 1:
                    for i in range(1, n):
                        self._set_reg(inst.a + i, VarargExpr())
                pc += 1

            # ===== Side-effect operations (emit statements) =====

            elif op == Op.SETGLOBAL:
                val = self._get_reg(inst.a)
                gname = self._global_name(inst.bx)

                # Pattern: CLOSURE + SETGLOBAL -> function name = ...
                # Just emit as global assignment
                stmts.append(AssignStmt([GlobalExpr(gname)], [val]))
                pc += 1

            elif op == Op.SETUPVAL:
                stmts.append(AssignStmt(
                    [UpvalExpr(self._upval_name(inst.b))],
                    [self._get_reg(inst.a)]))
                pc += 1

            elif op == Op.SETTABLE:
                table_expr = self._get_reg(inst.a)
                key = self._rk(inst.b)
                value = self._rk(inst.c)

                if inst.a in self.pending_tables:
                    self.pending_tables[inst.a].hash_part.append((key, value))
                else:
                    stmts.append(AssignStmt([IndexExpr(table_expr, key)], [value]))
                pc += 1

            elif op == Op.SETLIST:
                if inst.a in self.pending_tables:
                    tc = self.pending_tables[inst.a]
                    n = inst.b if inst.b != 0 else 0
                    for i in range(1, n + 1):
                        tc.array_part.append(self._get_reg(inst.a + i))
                pc += 1

            elif op == Op.CALL:
                # Finalize any pending table constructors before the call
                # to prevent table expressions from being inlined into call args
                for treg in list(self.pending_tables.keys()):
                    if treg != inst.a:  # don't finalize the function itself
                        tc = self.pending_tables.pop(treg)
                        name = self._reg_name(treg)
                        if treg not in self.local_declared:
                            self.locals[treg] = name
                            self.local_declared.add(treg)
                            stmts.append(LocalStmt([name], [tc]))
                        else:
                            stmts.append(AssignStmt([VarExpr(name)], [tc]))
                        self.regs[treg] = VarExpr(name)

                func_expr = self._get_reg(inst.a)
                nargs = inst.b - 1 if inst.b != 0 else -1
                nret = inst.c - 1 if inst.c != 0 else -1

                if nargs >= 0:
                    args = [self._get_reg(inst.a + 1 + i) for i in range(nargs)]
                else:
                    args = []
                    for i in range(1, self.proto.max_stack_size - inst.a):
                        r = inst.a + i
                        if r in self.regs:
                            args.append(self._get_reg(r))
                        else:
                            break
                    if not args:
                        args = [VarargExpr()]

                call_expr = FuncCallExpr(func_expr, args)

                if nret == 0:
                    stmts.append(CallStmt(call_expr))
                elif nret == 1:
                    self._set_reg(inst.a, call_expr)
                    # Always materialize call results to prevent expression explosion
                    self._maybe_materialize(inst.a, stmts)
                    # If not used later at all, emit as bare statement
                    if inst.a not in self.local_declared and not self._is_reg_used_later(inst.a, pc, end):
                        stmts.append(CallStmt(call_expr))
                        self.regs[inst.a] = VarExpr(self._reg_name(inst.a))
                else:
                    ret_count = nret if nret > 0 else 1
                    names = []
                    for i in range(ret_count):
                        r = inst.a + i
                        n = self._reg_name(r)
                        self.locals[r] = n
                        self.local_declared.add(r)
                        self.regs[r] = VarExpr(n)
                        names.append(n)
                    stmts.append(LocalStmt(names, [call_expr]))
                pc += 1

            elif op == Op.TAILCALL:
                func_expr = self._get_reg(inst.a)
                nargs = inst.b - 1 if inst.b != 0 else -1
                if nargs >= 0:
                    args = [self._get_reg(inst.a + 1 + i) for i in range(nargs)]
                else:
                    args = []
                stmts.append(ReturnStmt([FuncCallExpr(func_expr, args)]))
                pc += 1

            elif op == Op.RETURN:
                # Finalize pending tables before return
                for treg in list(self.pending_tables.keys()):
                    tc = self.pending_tables.pop(treg)
                    name = self._reg_name(treg)
                    if treg not in self.local_declared:
                        self.locals[treg] = name
                        self.local_declared.add(treg)
                        stmts.append(LocalStmt([name], [tc]))
                    else:
                        stmts.append(AssignStmt([VarExpr(name)], [tc]))
                    self.regs[treg] = VarExpr(name)

                nret = inst.b - 1 if inst.b != 0 else -1
                if nret == 0:
                    if pc != self.n - 1:  # skip implicit final return
                        stmts.append(ReturnStmt([]))
                elif nret > 0:
                    vals = [self._get_reg(inst.a + i) for i in range(nret)]
                    stmts.append(ReturnStmt(vals))
                else:
                    # Variable return (b==0): return R(a) to top
                    vals = [self._get_reg(inst.a)]
                    stmts.append(ReturnStmt(vals))
                pc += 1

            # ===== Control flow =====

            elif op == Op.JMP:
                target = pc + 1 + inst.sbx
                if target <= pc:
                    stmts.append(CommentStmt(f'loop back to PC {target}'))
                pc += 1

            elif op in (Op.EQ, Op.LT, Op.LE):
                self._materialize_all_complex(stmts)
                left = self._rk(inst.b)
                right = self._rk(inst.c)
                sym = COMPARE_SYMBOLS[op]

                if pc + 1 < end and self.code[pc + 1].op == Op.JMP:
                    jmp = self.code[pc + 1]
                    jump_target = pc + 2 + jmp.sbx

                    if inst.a == 0:
                        cond = BinopExpr(sym, left, right)
                    else:
                        neg = {'==': '~=', '<': '>=', '<=': '>'}[sym]
                        cond = BinopExpr(neg, left, right)

                    branch_stmts, next_pc = self._decompile_branch(cond, pc + 2, jump_target, end)
                    stmts.extend(branch_stmts)
                    pc = next_pc
                else:
                    stmts.append(CommentStmt(f'comparison {sym} at PC {pc}'))
                    pc += 1

            elif op == Op.TEST:
                self._materialize_all_complex(stmts)
                reg_expr = self._get_reg(inst.a)
                if pc + 1 < end and self.code[pc + 1].op == Op.JMP:
                    jmp = self.code[pc + 1]
                    jump_target = pc + 2 + jmp.sbx
                    cond = reg_expr if inst.c == 0 else UnopExpr('not', reg_expr)
                    branch_stmts, next_pc = self._decompile_branch(cond, pc + 2, jump_target, end)
                    stmts.extend(branch_stmts)
                    pc = next_pc
                else:
                    pc += 1

            elif op == Op.TESTSET:
                reg_expr = self._get_reg(inst.b)
                if pc + 1 < end and self.code[pc + 1].op == Op.JMP:
                    jmp = self.code[pc + 1]
                    jump_target = pc + 2 + jmp.sbx
                    cond = reg_expr if inst.c == 0 else UnopExpr('not', reg_expr)
                    self._set_reg(inst.a, reg_expr)
                    branch_stmts, next_pc = self._decompile_branch(cond, pc + 2, jump_target, end)
                    stmts.extend(branch_stmts)
                    pc = next_pc
                else:
                    self._set_reg(inst.a, self._get_reg(inst.b))
                    pc += 1

            elif op == Op.FORPREP:
                loop_body_start = pc + 1
                forloop_pc = pc + 1 + inst.sbx
                init_expr = self._get_reg(inst.a)
                limit_expr = self._get_reg(inst.a + 1)
                step_expr = self._get_reg(inst.a + 2)
                var_name = self._reg_name(inst.a + 3)
                self.locals[inst.a + 3] = var_name
                self.local_declared.add(inst.a + 3)
                self.regs[inst.a + 3] = VarExpr(var_name)
                body = self._decompile_range(loop_body_start, forloop_pc)
                stmts.append(ForNumStmt(var_name, init_expr, limit_expr, step_expr, body))
                pc = forloop_pc + 1

            elif op == Op.FORLOOP:
                pc += 1  # handled by FORPREP

            elif op == Op.TFORLOOP:
                # Generic for: iterator in regs a, a+1, a+2
                # Result vars start at a+3
                nvar = inst.c
                iter_expr = self._get_reg(inst.a)
                state_expr = self._get_reg(inst.a + 1)
                ctrl_expr = self._get_reg(inst.a + 2)

                var_names = []
                for i in range(nvar):
                    vn = self._reg_name(inst.a + 3 + i)
                    var_names.append(vn)
                    self.locals[inst.a + 3 + i] = vn
                    self.local_declared.add(inst.a + 3 + i)
                    self.regs[inst.a + 3 + i] = VarExpr(vn)

                # Next instruction should be JMP back
                if pc + 1 < end and self.code[pc + 1].op == Op.JMP:
                    jmp = self.code[pc + 1]
                    loop_start = pc + 2 + jmp.sbx
                    # The body is between loop_start and pc (TFORLOOP)
                    # But we need to find the real body...
                    # For now emit as generic for
                    iterators = [iter_expr, state_expr, ctrl_expr]
                    # Body was already decompiled (it's between the initial
                    # setup and this TFORLOOP). We'll emit the for-in header.
                    body = []  # TODO: extract body from earlier stmts
                    stmts.append(ForInStmt(var_names, iterators, body))
                    pc += 2
                else:
                    stmts.append(CommentStmt(f'TFORLOOP at PC {pc}'))
                    pc += 1

            elif op == Op.CLOSE:
                pc += 1

            else:
                stmts.append(CommentStmt(f'unknown opcode {op} at PC {pc}'))
                pc += 1

        return stmts

    def _decompile_branch(self, cond, body_start, jump_target, block_end):
        """Decompile if/elseif/else or while structure.
        Returns (stmts, next_pc)."""
        self._branch_depth = getattr(self, '_branch_depth', 0) + 1
        if self._branch_depth > 30:
            self._branch_depth -= 1
            return ([CommentStmt(f'nested branch (depth limit)')], jump_target)

        if jump_target > block_end:
            jump_target = block_end

        # Detect while loop: backward jump at end of body
        if jump_target > 0 and jump_target <= self.n:
            check_pc = jump_target - 1
            if check_pc < self.n and self.code[check_pc].op == Op.JMP:
                back_target = check_pc + 1 + self.code[check_pc].sbx
                if back_target <= body_start - 2:
                    body = self._decompile_range(body_start, check_pc)
                    self._branch_depth -= 1
                    return ([WhileStmt(cond, body)], jump_target)

        if_stmt = IfStmt()

        # Check for else block (JMP at end of then-body)
        if jump_target > body_start and jump_target <= self.n:
            last_body_pc = jump_target - 1
            if last_body_pc < self.n and self.code[last_body_pc].op == Op.JMP:
                else_jmp = self.code[last_body_pc]
                else_end = last_body_pc + 1 + else_jmp.sbx
                if else_end > jump_target:
                    body = self._decompile_range(body_start, last_body_pc)
                    if_stmt.branches.append((cond, body))

                    # Check for elseif chain
                    else_start_pc = jump_target
                    if else_start_pc < self.n:
                        first_else = self.code[else_start_pc]
                        if first_else.op in (Op.EQ, Op.LT, Op.LE, Op.TEST, Op.TESTSET):
                            # Potential elseif
                            else_body = self._decompile_range(jump_target, else_end)
                            if_stmt.branches.append((None, else_body))
                        else:
                            else_body = self._decompile_range(jump_target, else_end)
                            if_stmt.branches.append((None, else_body))
                    else:
                        else_body = self._decompile_range(jump_target, else_end)
                        if_stmt.branches.append((None, else_body))

                    self._branch_depth -= 1
                    return ([if_stmt], else_end)

        # Simple if (no else)
        body = self._decompile_range(body_start, jump_target)
        if_stmt.branches.append((cond, body))
        self._branch_depth -= 1
        return ([if_stmt], jump_target)


def format_function(proto, stmts, inline=False):
    """Format a decompiled function as Lua source."""
    params = []
    for i in range(proto.num_params):
        # Try to get param name from locvars
        name = None
        for lv_name, start, end in proto.locvars:
            if start == 0 and lv_name:
                name = lv_name
                break
        if not name:
            name = f'a{i}'
        params.append(name)

    if proto.is_vararg & 0x02:
        params.append('...')

    param_str = ', '.join(params)
    header = f'function({param_str})'

    body_lines = []
    for s in stmts:
        body_lines.append(s.to_lua(1))

    if inline:
        if not body_lines:
            return f'{header} end'
        if len(body_lines) == 1 and len(body_lines[0]) < 60:
            return f'{header} {body_lines[0].strip()} end'

    lines = [header]
    lines.extend(body_lines)
    lines.append('end')
    return '\n'.join(lines)


def decompile_proto(proto, out, depth=0, func_id='0', input_file=''):
    """Recursively decompile a function prototype and write to output."""
    decompiler = Decompiler(proto, proto_idx=int(func_id.replace('_', '')) if func_id.isdigit() else 0)
    try:
        stmts = decompiler.decompile()
    except RecursionError:
        stmts = [CommentStmt('decompiler hit recursion limit — try --disasm mode')]

    if depth == 0:
        source = proto.source or input_file or '(unknown)'
        out.write(f'-- Decompiled using jinzear v1.0 by 0xmadvise\n')
        out.write(f'-- https://github.com/0xmadvise/jinzear\n')
        out.write(f'--\n')
        out.write(f'-- Source: {source}\n')
        out.write(f'-- Params: {proto.num_params}{"+" if proto.is_vararg else ""}, ')
        out.write(f'{proto.max_stack_size} slots, {proto.num_upvalues} upvalues, ')
        out.write(f'{len(proto.constants)} constants, {len(proto.protos)} functions\n')
        out.write(f'\n')
        for s in stmts:
            out.write(s.to_lua(0))
            out.write('\n')
    else:
        text = format_function(proto, stmts, inline=True)
        out.write(text)


def extract_strings(data):
    """Extract all decrypted string constants from a Fate/Z or Lua file."""
    loader = Loader(data)
    proto = loader.load()
    strings = []

    def walk(p, path=''):
        for i, c in enumerate(p.constants):
            if c.type == 'string':
                val = c.value
                if isinstance(val, bytes):
                    val = val.decode('utf-8', errors='replace')
                strings.append((path, val))
        for i, child in enumerate(p.protos):
            walk(child, f'{path}_{i}' if path else str(i))

    walk(proto)
    return strings


class Lua51Writer:
    """Writes standard Lua 5.1 bytecode for format conversion mode."""
    def __init__(self, sizeof_size_t=4):
        self.buf = io.BytesIO()
        self.sizeof_size_t = sizeof_size_t

    def write_byte(self, val):
        self.buf.write(bytes([val]))

    def write_bytes(self, data):
        self.buf.write(data)

    def write_int(self, val):
        self.buf.write(struct.pack('<I', val))

    def write_size_t(self, val):
        if self.sizeof_size_t == 8:
            self.buf.write(struct.pack('<Q', val))
        else:
            self.buf.write(struct.pack('<I', val))

    def write_number(self, data):
        self.buf.write(data)

    def write_string(self, size, data):
        self.write_size_t(size)
        if size > 0 and data is not None:
            self.buf.write(data)

    def getvalue(self):
        return self.buf.getvalue()


def convert_to_lua51(input_data, native=False):
    """Convert Fate/Z bytecode to standard Lua 5.1 bytecode (for use with unluac/luadec)."""
    if input_data[:8] != FATEZ_MAGIC:
        raise ValueError("Not a Fate/Z file")

    reader = BytecodeReader(input_data)
    reader.off = 8
    version = reader.u8()
    fmt = reader.u8()
    endian = reader.u8()
    sizeof_int = reader.u8()
    sizeof_size_t = reader.u8()
    sizeof_instr = reader.u8()
    sizeof_number = reader.u8()
    extra = reader.u8()
    reader.sizeof_size_t = sizeof_size_t

    target_size_t = struct.calcsize('P') if native else sizeof_size_t
    writer = Lua51Writer(sizeof_size_t=target_size_t)

    # Standard header
    writer.write_bytes(LUA51_MAGIC)
    writer.write_byte(0x51)
    writer.write_byte(0x00)
    writer.write_byte(endian)
    writer.write_byte(sizeof_int)
    writer.write_byte(target_size_t)
    writer.write_byte(sizeof_instr)
    writer.write_byte(sizeof_number)
    writer.write_byte(0x00)

    _convert_function(reader, writer, 0, 0)

    if reader.remaining() > 0:
        sys.stderr.write(f"Warning: {reader.remaining()} trailing bytes\n")

    return writer.getvalue()


def _convert_function(reader, writer, depth, parent_nups):
    """Convert one function prototype from Fate/Z to standard Lua 5.1 bytecode."""
    nups_raw = reader.u8()
    src_size = reader.u32()
    numparams = reader.u8()
    linedef = reader.u32()
    is_vararg = reader.u8()
    lastlinedef = reader.u32()
    maxstack = reader.u8()

    nups = parent_nups if (depth > 0 and nups_raw == 0) else nups_raw

    writer.write_string(0, None)
    writer.write_int(linedef)
    writer.write_int(lastlinedef)
    writer.write_byte(nups)
    writer.write_byte(numparams)
    writer.write_byte(is_vararg)
    writer.write_byte(maxstack)

    # Instructions
    sizecode = reader.u32()
    writer.write_int(sizecode)
    instructions = []
    for _ in range(sizecode):
        raw = reader.u32()
        instructions.append(raw)
        # Remap FZ opcode
        fz_op = raw & 0x3F
        if fz_op != 0 and fz_op in FZ_TO_STD:
            std_op = FZ_TO_STD[fz_op]
            new = (raw & ~0x3F) | std_op
            if fz_op == 2:  # LEN: clear C
                new = new & ~(0x1FF << 14)
            writer.write_int(new)
        else:
            writer.write_int(raw)

    # Constants
    numk = reader.u32()
    writer.write_int(numk)
    for _ in range(numk):
        t = reader.u8()
        base_t = t & 0x7F
        if base_t in (FATEZ_NIL, FATEZ_NIL_ALT):
            writer.write_byte(STD_NIL)
        elif base_t == FATEZ_BOOL:
            val = reader.u8()
            writer.write_byte(STD_BOOL)
            writer.write_byte(val)
        elif base_t == FATEZ_NUMBER:
            data = reader.raw(8)
            writer.write_byte(STD_NUMBER)
            writer.write_bytes(data)
        elif base_t == FATEZ_STRING:
            ssize = reader.u32()
            enc = reader.raw(ssize) if ssize > 0 else b''
            dec = decrypt_fatez_string(enc) if enc else enc
            writer.write_byte(STD_STRING)
            writer.write_string(ssize, dec)
        elif base_t == FATEZ_INT:
            int_data = reader.raw(4)
            val = struct.unpack('<i', int_data)[0]
            double_data = struct.pack('<d', float(val))
            writer.write_byte(STD_NUMBER)
            writer.write_bytes(double_data)
        else:
            raise ValueError(f"Unknown Fate/Z const type 0x{t:02x}")

    # Infer child nups
    proto_nups = []
    i = 0
    while i < len(instructions):
        fz_op = instructions[i] & 0x3F
        if fz_op == 1:  # CLOSURE
            n = 0
            j = i + 1
            while j < len(instructions):
                next_op = instructions[j] & 0x3F
                if next_op in (36, 18):  # MOVE or GETUPVAL
                    n += 1
                    j += 1
                else:
                    break
            proto_nups.append(n)
            i = j
        else:
            i += 1

    # Nested protos
    num_protos = reader.u32()
    writer.write_int(num_protos)
    for pi in range(num_protos):
        child_nups = proto_nups[pi] if pi < len(proto_nups) else 0
        _convert_function(reader, writer, depth + 1, child_nups)

    # Debug: lineinfo
    n_lineinfo = reader.u32()
    writer.write_int(n_lineinfo)
    for _ in range(n_lineinfo):
        writer.write_int(reader.u32())

    # Debug: locvars
    n_locvars = reader.u32()
    writer.write_int(n_locvars)
    for _ in range(n_locvars):
        reader.u32()  # name size only
        writer.write_string(0, None)
        writer.write_int(reader.u32())  # startpc
        writer.write_int(reader.u32())  # endpc

    # Debug: upvalue names
    n_upvalues = reader.u32()
    writer.write_int(n_upvalues)
    for _ in range(n_upvalues):
        reader.u32()  # name size only
        writer.write_string(0, None)


def process_file(input_path, output_path, mode='decompile', native=False):
    """Process a single bytecode file."""
    with open(input_path, 'rb') as f:
        data = f.read()

    is_fatez = data[:8] == FATEZ_MAGIC
    is_lua51 = data[:4] == LUA51_MAGIC

    if not is_fatez and not is_lua51:
        sys.stderr.write(f"Skipping {input_path}: not Lua bytecode\n")
        return False

    try:
        if mode == 'convert':
            if not is_fatez:
                sys.stderr.write(f"Skipping {input_path}: not Fate/Z (already standard)\n")
                return False
            result = convert_to_lua51(data, native=native)
            if output_path:
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(result)
            else:
                sys.stdout.buffer.write(result)
            return True

        elif mode == 'disasm':
            loader = Loader(data)
            proto = loader.load()
            out = io.StringIO()
            disassemble_proto(proto, out)
            text = out.getvalue()
            if output_path:
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(text)
            else:
                print(text)
            return True

        elif mode == 'strings':
            strings = extract_strings(data)
            for path, val in strings:
                print(f'[{path}] {val}')
            return True

        else:  # decompile
            loader = Loader(data)
            proto = loader.load()
            out = io.StringIO()
            decompile_proto(proto, out, input_file=os.path.basename(input_path))
            text = out.getvalue()
            if output_path:
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(text)
            else:
                print(text)
            return True

    except RecursionError:
        sys.stderr.write(f"Error processing {input_path}: recursion limit (try --disasm)\n")
        return False
    except Exception as e:
        sys.stderr.write(f"Error processing {input_path}: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def process_directory(input_dir, output_dir, mode='decompile', native=False):
    """Process all bytecode files in a directory tree."""
    success = fail = skip = 0
    ext = '.lua' if mode == 'decompile' else '.luac' if mode == 'convert' else '.dis'

    for dirpath, dirnames, filenames in os.walk(input_dir):
        for fname in filenames:
            if not fname.endswith('.lua') and not fname.endswith('.luac'):
                continue
            input_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(input_path, input_dir)

            with open(input_path, 'rb') as f:
                magic = f.read(8)

            if magic[:8] != FATEZ_MAGIC and magic[:4] != LUA51_MAGIC:
                skip += 1
                continue

            if mode == 'decompile':
                out_name = rel_path
            elif mode == 'convert':
                out_name = rel_path
            else:
                out_name = rel_path + '.dis'

            output_path = os.path.join(output_dir, out_name)

            if process_file(input_path, output_path, mode=mode, native=native):
                success += 1
                print(f'  OK: {rel_path}')
            else:
                fail += 1

    return success, fail, skip


def main():
    sys.setrecursionlimit(50000)

    parser = argparse.ArgumentParser(
        prog='jinzear',
        description='Fate/Z Lua Bytecode Decompiler -- full self-contained '
                    'decompiler for MiWiFi firmware Lua bytecode. '
                    'Supports both Fate/Z and standard Lua 5.1 bytecode.',
        epilog='Author: 0xmadvise | https://github.com/0xmadvise/jinzear'
    )
    parser.add_argument('input', help='Input file or directory')
    parser.add_argument('-o', '--output', help='Output file or directory')

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('-d', '--decompile', action='store_true', default=True,
                           help='Decompile to Lua source (default)')
    mode_group.add_argument('-D', '--disasm', action='store_true',
                           help='Disassemble (luac -l style)')
    mode_group.add_argument('-c', '--convert', action='store_true',
                           help='Convert Fate/Z to standard Lua 5.1 bytecode')
    mode_group.add_argument('-s', '--strings', action='store_true',
                           help='Extract string constants only')

    parser.add_argument('--batch', action='store_true',
                       help='Process entire directory tree')
    parser.add_argument('--native', action='store_true',
                       help='Use host sizeof(size_t) in converted bytecode')

    args = parser.parse_args()

    if args.disasm:
        mode = 'disasm'
    elif args.convert:
        mode = 'convert'
    elif args.strings:
        mode = 'strings'
    else:
        mode = 'decompile'

    if args.batch or os.path.isdir(args.input):
        if not args.output:
            print("Error: -o/--output directory required for batch processing")
            sys.exit(1)
        print(f'jinzear: Processing {args.input} -> {args.output} (mode: {mode})')
        s, f, sk = process_directory(args.input, args.output, mode=mode, native=args.native)
        print(f'\nResults: {s} converted, {f} failed, {sk} skipped')
    else:
        output = args.output
        if not output and mode in ('decompile', 'disasm'):
            output = None  # stdout
        elif not output and mode == 'convert':
            output = args.input + '.std'

        if process_file(args.input, output, mode=mode, native=args.native):
            if output:
                print(f'jinzear: {args.input} -> {output}', file=sys.stderr)
        else:
            sys.exit(1)


if __name__ == '__main__':
    main()
