1;  % Octave side of run_v1.py (study P001, cycle 1); a script file, not a function file.
% Calls the authors' psnrhvsm.m and HaarPSI.m (read from <metric_sources>, never copied here)
% on uint8 pairs written by run_v1.py.
% Usage: octave-cli --no-gui --quiet --norc octave_v1.m <metric_sources> <io_dir>
% <io_dir>/list.txt: one line "k H W" per pair; r<k>.bin, d<k>.bin: uint8, H x W, row-major.
% Writes <io_dir>/out.txt: one line "k psnrhvsm haarpsi" per pair, %.17g.

function X = read_u8(path, h, w)
  f = fopen(path, 'rb');
  X = fread(f, [w h], '*uint8')';
  fclose(f);
end

args = argv();
src = args{1};
io = args{2};
pkg load signal;  % dct2 for psnrhvsm.m
% psnrhvsm.m uses `|` inside `if` (lines 136 and 181): Octave warns on every block; harmless.
warning('off', 'Octave:possible-matlab-short-circuit-operator');
addpath(fullfile(src, 'psnrhvsm'));
addpath(fullfile(src, 'haarpsi'));
L = load(fullfile(io, 'list.txt'));
out = fopen(fullfile(io, 'out.txt'), 'w');
for i = 1:rows(L)
  k = L(i, 1); h = L(i, 2); w = L(i, 3);
  A = read_u8(fullfile(io, sprintf('r%d.bin', k)), h, w);
  B = read_u8(fullfile(io, sprintf('d%d.bin', k)), h, w);
  fprintf(out, '%d %.17g %.17g\n', k, psnrhvsm(A, B), HaarPSI(A, B));
end
fclose(out);
