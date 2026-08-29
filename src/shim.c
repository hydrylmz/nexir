#include <stdint.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
#include <libavutil/channel_layout.h>
#include <libavutil/pixfmt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/mastering_display_metadata.h>

int get_av_pix_fmt_rgbaf16le() {
#ifdef AV_PIX_FMT_RGBAF16LE
    return AV_PIX_FMT_RGBAF16LE;
#else
    return -1;
#endif
}

void avcodec_ctx_set_time_base(AVCodecContext* ctx, AVRational tb) { ctx->time_base = tb; }
void avcodec_ctx_set_flags(AVCodecContext* ctx, int flags) { ctx->flags |= flags; }
void avcodec_ctx_set_dimensions(AVCodecContext* ctx, int w, int h) { ctx->width = w; ctx->height = h; }
void avcodec_ctx_set_pix_fmt(AVCodecContext* ctx, enum AVPixelFormat fmt) { ctx->pix_fmt = fmt; }
void avcodec_ctx_set_gop_size(AVCodecContext* ctx, int gop) { ctx->gop_size = gop; }
void avcodec_ctx_set_bit_rate(AVCodecContext* ctx, int64_t br) { ctx->bit_rate = br; }
void avcodec_ctx_set_sample_rate(AVCodecContext* ctx, int sr) { ctx->sample_rate = sr; }
void avcodec_ctx_set_ch_layout(AVCodecContext* ctx, uint64_t ch_mask) { av_channel_layout_from_mask(&ctx->ch_layout, ch_mask); }
void avcodec_ctx_set_sample_fmt(AVCodecContext* ctx, int fmt) { ctx->sample_fmt = fmt; }
void avcodec_set_hw_device_ctx(AVCodecContext* ctx, AVBufferRef* hw) { ctx->hw_device_ctx = hw; }
void avcodec_set_thread_count(AVCodecContext* ctx, int c) { ctx->thread_count = c; }

static enum AVPixelFormat default_get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *fmt) {
    (void)ctx;
    for (const enum AVPixelFormat *p = fmt; *p != -1; p++) {
        if (*p == AV_PIX_FMT_CUDA || *p == AV_PIX_FMT_D3D11 || *p == AV_PIX_FMT_D3D11VA_VLD || *p == AV_PIX_FMT_NV12) {
            return *p;
        }
    }
    return fmt[0];
}

void avcodec_enable_hw_get_format(AVCodecContext* ctx) {
    ctx->get_format = default_get_hw_format;
}

const char* avcodec_get_name_shim(const AVCodec* c) { return c->name; }

AVStream* avformat_get_stream(AVFormatContext* ctx, int idx) { return ctx->streams[idx]; }
AVRational avstream_get_time_base(AVStream* st) { return st->time_base; }
void avstream_set_time_base(AVStream* st, AVRational tb) { st->time_base = tb; }
AVRational avstream_get_avg_frame_rate(AVStream* st) { return st->avg_frame_rate; }
AVRational avstream_get_r_frame_rate(AVStream* st) { return st->r_frame_rate; }
int64_t avformat_get_duration(AVFormatContext* ctx) { return ctx->duration; }
AVCodecParameters* avstream_get_codecpar(AVStream* st) { return st->codecpar; }
AVCodecParameters* avstream_get_codecpar_mut(AVStream* st) { return st->codecpar; }

enum AVCodecID avcodecpar_get_codec_id(AVCodecParameters* par) { return par->codec_id; }
int avcodecpar_get_width(AVCodecParameters* par) { return par->width; }
int avcodecpar_get_height(AVCodecParameters* par) { return par->height; }
int avcodecpar_get_color_space(AVCodecParameters* par) { return par->color_space; }
int avcodecpar_get_color_range(AVCodecParameters* par) { return par->color_range; }
int avcodecpar_get_color_trc(AVCodecParameters* par) { return par->color_trc; }
int avcodecpar_get_color_primaries(AVCodecParameters* par) { return par->color_primaries; }
int avcodecpar_get_bit_depth(AVCodecParameters* par) {
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(par->format);
    if (desc) {
        return desc->comp[0].depth;
    }
    return 8; // fallback
}

int av_packet_stream_index(AVPacket* pkt) { return pkt->stream_index; }
int64_t av_packet_pts(AVPacket* pkt) { return pkt->pts; }
int64_t av_packet_duration(AVPacket* pkt) { return pkt->duration; }

int av_frame_get_width(AVFrame* f) { return f->width; }
int av_frame_get_height(AVFrame* f) { return f->height; }
int64_t av_frame_get_pts(AVFrame* f) { return f->pts; }
void av_frame_set_pts(AVFrame* f, int64_t pts) { f->pts = pts; }
/* Frame duration, in the frame's own timebase (the encoder's time_base for
 * frames handed to avcodec_send_frame).  libavcodec copies it onto the output
 * AVPacket, and the mp4 muxer needs a non-zero duration on the LAST packet to
 * size the final `stts` entry — without it the track's mdhd duration stops at
 * the final frame's PTS and decoders discard that frame (AV_PKT_FLAG_DISCARD). */
void av_frame_set_duration(AVFrame* f, int64_t duration) { f->duration = duration; }
int* av_frame_get_linesize(AVFrame* f) { return f->linesize; }
int av_frame_get_format(AVFrame* f) { return f->format; }
int av_frame_get_nb_samples(AVFrame* f) { return f->nb_samples; }
uint8_t** av_frame_get_data(AVFrame* f) { return f->data; }
int av_frame_get_sample_rate(AVFrame* f) { return f->sample_rate; }
int av_frame_get_color_space(AVFrame* f) { return f->colorspace; }
int av_frame_get_color_range(AVFrame* f) { return f->color_range; }
int av_frame_get_color_trc(AVFrame* f) { return f->color_trc; }
int av_frame_get_color_primaries(AVFrame* f) { return f->color_primaries; }

/* Colour-property setters.
 *
 * Needed because av_hwframe_transfer_data copies PIXELS only: the destination
 * frame keeps its default "unspecified" colour fields, which would send every
 * hardware-decoded HDR frame down the SDR path.  The decoder copies the
 * properties across explicitly after the transfer. */
void av_frame_set_color_space(AVFrame* f, int v) { f->colorspace = (enum AVColorSpace)v; }
void av_frame_set_color_range(AVFrame* f, int v) { f->color_range = (enum AVColorRange)v; }
void av_frame_set_color_trc(AVFrame* f, int v) { f->color_trc = (enum AVColorTransferCharacteristic)v; }
void av_frame_set_color_primaries(AVFrame* f, int v) { f->color_primaries = (enum AVColorPrimaries)v; }

/* Bits per component of a pixel format, from FFmpeg's own descriptor table.
 * Returns 0 for formats with no descriptor (hardware surfaces), which the caller
 * treats as "unknown". */
int av_pix_fmt_bit_depth(int fmt) {
    const AVPixFmtDescriptor* d = av_pix_fmt_desc_get((enum AVPixelFormat)fmt);
    return d ? d->comp[0].depth : 0;
}

/* Number of planes in a pixel format; 0 when the format has no descriptor. */
int av_pix_fmt_plane_count(int fmt) {
    const AVPixFmtDescriptor* d = av_pix_fmt_desc_get((enum AVPixelFormat)fmt);
    return d ? av_pix_fmt_count_planes((enum AVPixelFormat)fmt) : 0;
}

/* Bit shift of component 0 inside its storage word.  P010 stores 10-bit codes
 * MSB-aligned in 16-bit words and reports shift=6; every planar high-depth
 * format is LSB-aligned and reports 0.  That distinction is a factor of 64 in
 * brightness if it is guessed wrong. */
int av_pix_fmt_component_shift(int fmt) {
    const AVPixFmtDescriptor* d = av_pix_fmt_desc_get((enum AVPixelFormat)fmt);
    return d ? d->comp[0].shift : 0;
}

/* AVCodecContext colour properties — set on an ENCODER context before
 * avcodec_open2 so the muxer copies them into the container (mp4 `colr` box,
 * Matroska colour element).  Without them an HDR export decodes as SDR no
 * matter what the bitstream contains. */
void avcodec_ctx_set_color_space(AVCodecContext* ctx, int v) { ctx->colorspace = (enum AVColorSpace)v; }
void avcodec_ctx_set_color_range(AVCodecContext* ctx, int v) { ctx->color_range = (enum AVColorRange)v; }
void avcodec_ctx_set_color_trc(AVCodecContext* ctx, int v) { ctx->color_trc = (enum AVColorTransferCharacteristic)v; }
void avcodec_ctx_set_color_primaries(AVCodecContext* ctx, int v) { ctx->color_primaries = (enum AVColorPrimaries)v; }
void avcodec_ctx_set_chroma_location(AVCodecContext* ctx, int v) { ctx->chroma_sample_location = (enum AVChromaLocation)v; }

int avcodec_ctx_get_color_space(AVCodecContext* ctx) { return ctx->colorspace; }
int avcodec_ctx_get_color_range(AVCodecContext* ctx) { return ctx->color_range; }
int avcodec_ctx_get_color_trc(AVCodecContext* ctx) { return ctx->color_trc; }
int avcodec_ctx_get_color_primaries(AVCodecContext* ctx) { return ctx->color_primaries; }

/* AVCodecParameters colour properties — written directly on the muxer's stream
 * so container metadata is correct even when the stream was described by a
 * parameter-only encoder context (the NVENC path does exactly that). */
void avcodecpar_set_color_space(AVCodecParameters* par, int v) { par->color_space = (enum AVColorSpace)v; }
void avcodecpar_set_color_range(AVCodecParameters* par, int v) { par->color_range = (enum AVColorRange)v; }
void avcodecpar_set_color_trc(AVCodecParameters* par, int v) { par->color_trc = (enum AVColorTransferCharacteristic)v; }
void avcodecpar_set_color_primaries(AVCodecParameters* par, int v) { par->color_primaries = (enum AVColorPrimaries)v; }
void avcodecpar_set_chroma_location(AVCodecParameters* par, int v) { par->chroma_location = (enum AVChromaLocation)v; }

/* ── P1.7: HDR10 static metadata ──────────────────────────────────────────────
 *
 * SMPTE ST 2086 mastering-display colour volume (MDCV) and CTA-861.3 content
 * light level (CLL).  Colour PRIMARIES/TRC tags alone tell a display "this is
 * BT.2020 PQ"; these two tell it what the content was actually graded on, which
 * is what a real HDR10 file must carry and what `ffprobe -show_frames` reports
 * as `mastering_display_metadata` / `content_light_level`.
 *
 * Units are the ones the standard uses, so the caller passes integers and no
 * float ever crosses the FFI:
 *   - chromaticity x/y in increments of 0.00002  (denominator 50000)
 *   - luminance in increments of 0.0001 cd/m^2   (denominator 10000)
 *
 * `prim` is 6 numerators in R.x R.y G.x G.y B.x B.y order; `wp` is 2.
 *
 * Two destinations are needed and they are NOT interchangeable:
 *   1. AVCodecContext::decoded_side_data — read by the ENCODER at
 *      avcodec_open2 time, which is what makes libx265/x264 emit the SEI
 *      messages inside the bitstream.
 *   2. AVCodecParameters::coded_side_data — read by the MUXER, which is what
 *      writes the mp4 `mdcv`/`clli` boxes (and the Matroska colour elements).
 * A file with only (1) loses its metadata to any remux; a file with only (2)
 * loses it to any stream copy into a container that has no such boxes. */

#define NEXIR_MDCV_CHROMA_DEN 50000
#define NEXIR_MDCV_LUMA_DEN   10000

static void nexir_fill_mdcv(AVMasteringDisplayMetadata* m,
                            const int* prim, const int* wp,
                            int min_luminance, int max_luminance) {
    for (int i = 0; i < 3; i++) {
        m->display_primaries[i][0].num = prim[i * 2];
        m->display_primaries[i][0].den = NEXIR_MDCV_CHROMA_DEN;
        m->display_primaries[i][1].num = prim[i * 2 + 1];
        m->display_primaries[i][1].den = NEXIR_MDCV_CHROMA_DEN;
    }
    m->white_point[0].num = wp[0];
    m->white_point[0].den = NEXIR_MDCV_CHROMA_DEN;
    m->white_point[1].num = wp[1];
    m->white_point[1].den = NEXIR_MDCV_CHROMA_DEN;
    m->min_luminance.num = min_luminance;
    m->min_luminance.den = NEXIR_MDCV_LUMA_DEN;
    m->max_luminance.num = max_luminance;
    m->max_luminance.den = NEXIR_MDCV_LUMA_DEN;
    m->has_primaries = 1;
    m->has_luminance = 1;
}

/* Attach MDCV + CLL to an encoder context.  Must be called BEFORE
 * avcodec_open2: libavcodec reads decoded_side_data there and takes ownership
 * of the array afterwards. Returns 0 on success, -1 on allocation failure. */
int avcodec_ctx_set_hdr10_metadata(AVCodecContext* ctx,
                                   const int* prim, const int* wp,
                                   int min_luminance, int max_luminance,
                                   unsigned max_cll, unsigned max_fall) {
    AVFrameSideData* sd = av_frame_side_data_new(
        &ctx->decoded_side_data, &ctx->nb_decoded_side_data,
        AV_FRAME_DATA_MASTERING_DISPLAY_METADATA,
        sizeof(AVMasteringDisplayMetadata),
        AV_FRAME_SIDE_DATA_FLAG_REPLACE);
    if (!sd) return -1;
    nexir_fill_mdcv((AVMasteringDisplayMetadata*)sd->data, prim, wp,
                    min_luminance, max_luminance);

    AVFrameSideData* cl = av_frame_side_data_new(
        &ctx->decoded_side_data, &ctx->nb_decoded_side_data,
        AV_FRAME_DATA_CONTENT_LIGHT_LEVEL,
        sizeof(AVContentLightMetadata),
        AV_FRAME_SIDE_DATA_FLAG_REPLACE);
    if (!cl) return -1;
    ((AVContentLightMetadata*)cl->data)->MaxCLL  = max_cll;
    ((AVContentLightMetadata*)cl->data)->MaxFALL = max_fall;
    return 0;
}

/* Attach MDCV + CLL to a muxer stream's codec parameters.  Must be called
 * BEFORE avformat_write_header. Returns 0 on success, -1 on failure. */
int avcodecpar_set_hdr10_metadata(AVCodecParameters* par,
                                  const int* prim, const int* wp,
                                  int min_luminance, int max_luminance,
                                  unsigned max_cll, unsigned max_fall) {
    AVPacketSideData* sd = av_packet_side_data_new(
        &par->coded_side_data, &par->nb_coded_side_data,
        AV_PKT_DATA_MASTERING_DISPLAY_METADATA,
        sizeof(AVMasteringDisplayMetadata), 0);
    if (!sd) return -1;
    nexir_fill_mdcv((AVMasteringDisplayMetadata*)sd->data, prim, wp,
                    min_luminance, max_luminance);

    AVPacketSideData* cl = av_packet_side_data_new(
        &par->coded_side_data, &par->nb_coded_side_data,
        AV_PKT_DATA_CONTENT_LIGHT_LEVEL,
        sizeof(AVContentLightMetadata), 0);
    if (!cl) return -1;
    ((AVContentLightMetadata*)cl->data)->MaxCLL  = max_cll;
    ((AVContentLightMetadata*)cl->data)->MaxFALL = max_fall;
    return 0;
}

/* Read HDR10 static metadata back off a demuxed stream, for verification.
 *
 * `out` receives 8 values:
 *   [0] has_primaries      [1] has_luminance
 *   [2] min_luminance num  [3] min_luminance den
 *   [4] max_luminance num  [5] max_luminance den
 *   [6] MaxCLL             [7] MaxFALL
 *
 * Returns a bitmask: 1 = MDCV present, 2 = CLL present, 0 = neither. */
int avcodecpar_get_hdr10_metadata(const AVCodecParameters* par, int64_t* out) {
    int found = 0;
    for (int i = 0; i < 8; i++) out[i] = 0;

    const AVPacketSideData* sd = av_packet_side_data_get(
        par->coded_side_data, par->nb_coded_side_data,
        AV_PKT_DATA_MASTERING_DISPLAY_METADATA);
    if (sd && sd->size >= sizeof(AVMasteringDisplayMetadata)) {
        const AVMasteringDisplayMetadata* m =
            (const AVMasteringDisplayMetadata*)sd->data;
        out[0] = m->has_primaries;
        out[1] = m->has_luminance;
        out[2] = m->min_luminance.num;
        out[3] = m->min_luminance.den;
        out[4] = m->max_luminance.num;
        out[5] = m->max_luminance.den;
        found |= 1;
    }

    const AVPacketSideData* cl = av_packet_side_data_get(
        par->coded_side_data, par->nb_coded_side_data,
        AV_PKT_DATA_CONTENT_LIGHT_LEVEL);
    if (cl && cl->size >= sizeof(AVContentLightMetadata)) {
        const AVContentLightMetadata* c = (const AVContentLightMetadata*)cl->data;
        out[6] = c->MaxCLL;
        out[7] = c->MaxFALL;
        found |= 2;
    }

    return found;
}


SwrContext* swr_alloc_set_opts(SwrContext* s, int64_t out_ch_layout, enum AVSampleFormat out_sample_fmt, int out_sample_rate, int64_t in_ch_layout, enum AVSampleFormat in_sample_fmt, int in_sample_rate, int log_offset, void* log_ctx) {
    AVChannelLayout out_layout;
    AVChannelLayout in_layout;
    av_channel_layout_from_mask(&out_layout, out_ch_layout);
    av_channel_layout_from_mask(&in_layout, in_ch_layout);
    if (swr_alloc_set_opts2(&s, &out_layout, out_sample_fmt, out_sample_rate, &in_layout, in_sample_fmt, in_sample_rate, log_offset, log_ctx) < 0) {
        return NULL;
    }
    return s;
}

int avcodec_ctx_get_sample_rate(AVCodecContext* ctx) { return ctx->sample_rate; }
int avcodec_ctx_get_channels(AVCodecContext* ctx) { return ctx->ch_layout.nb_channels; }
int avcodec_ctx_get_sample_fmt(AVCodecContext* ctx) { return ctx->sample_fmt; }
uint64_t avcodec_ctx_get_channel_layout(AVCodecContext* ctx) { return ctx->ch_layout.u.mask; }

void av_frame_set_width(AVFrame* f, int w) { f->width = w; }
void av_frame_set_height(AVFrame* f, int h) { f->height = h; }
void av_frame_set_format(AVFrame* f, int fmt) { f->format = fmt; }

void av_frame_set_nb_samples(AVFrame* f, int samples) { f->nb_samples = samples; }
void av_frame_set_sample_rate(AVFrame* f, int rate) { f->sample_rate = rate; }
void av_frame_set_ch_layout(AVFrame* f, uint64_t ch_mask) {
    av_channel_layout_from_mask(&f->ch_layout, ch_mask);
}

int av_stream_get_index(AVStream* st) { return st->index; }

int avformat_open_output_pb(AVFormatContext* s, const char* url, int flags) {
    return avio_open(&s->pb, url, flags);
}

void av_packet_set_stream_index(AVPacket* pkt, int idx) {
    pkt->stream_index = idx;
}

int avformat_write_header_shim(AVFormatContext *s, AVDictionary **options) {
    for (int i = 0; i < s->nb_streams; i++) {
        AVStream* st = s->streams[i];
        fprintf(stderr, "[shim] Stream %d: codec_type=%d, codec_id=%d, w=%d, h=%d, sr=%d, channels=%d, extradata_size=%d, time_base=%d/%d\n",
            i, st->codecpar->codec_type, st->codecpar->codec_id,
            st->codecpar->width, st->codecpar->height,
            st->codecpar->sample_rate, st->codecpar->ch_layout.nb_channels,
            st->codecpar->extradata_size,
            st->time_base.num, st->time_base.den);
    }
    int ret = avformat_write_header(s, options);
    if (ret < 0) {
        char errbuf[128];
        av_strerror(ret, errbuf, sizeof(errbuf));
        fprintf(stderr, "[shim] avformat_write_header failed: %d (%s)\n", ret, errbuf);
    }
    return ret;
}
