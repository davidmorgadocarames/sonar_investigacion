"""Signal parameters approved in D-015. Every value used by preprocessing and features lives here."""

# Input format (F1-T03 inventory; checked by qc).
SAMPLE_RATE = 32000

# Preprocessing: DC offset removed per recording (D-009), then fixed-length segments without
# overlap (the tail that does not fill a segment is dropped), each scaled to unit RMS.
SEGMENT_S = 3.0

# LOFAR: decimate to 5 kHz (D-015 adjustment: the anti-aliasing transition band stays above
# 2 kHz), Hann window of 0.5 s with a 0.25 s hop (2 Hz bins), keep 0-2 kHz, TPSW background
# normalisation, output in dB above the background.
LOFAR_SAMPLE_RATE = 5000
LOFAR_WINDOW_S = 0.5
LOFAR_HOP_S = 0.25
LOFAR_MAX_HZ = 2000.0
TPSW_HALF_WINDOW_BINS = 20
TPSW_GAP_BINS = 2
TPSW_CLIP_FACTOR = 2.0

# DEMON: band-pass 2-8 kHz (Butterworth, zero phase), |Hilbert| envelope minus its mean,
# decimated to 500 Hz in two x8 stages, Hann spectrum of the whole segment (0-100 Hz),
# bins below 1 Hz set to 0, normalised by its maximum.
DEMON_BAND_HZ = (2000.0, 8000.0)
DEMON_FILTER_ORDER = 4
DEMON_DECIMATION = (8, 8)
DEMON_MAX_HZ = 100.0
DEMON_MIN_HZ = 1.0
