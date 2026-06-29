#include <stdint.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
#include <libavutil/channel_layout.h>

void avcodec_ctx_set_time_base(AVCodecContext* ctx, AVRational tb) { ctx->time_base = tb; }
void avcodec_ctx_set_flags(AVCodecContext* ctx, int flags) { ctx->flags = flags; }
void avcodec_ctx_set_dimensions(AVCodecContext* ctx, int w, int h) { ctx->width = w; ctx->height = h; }
void avcodec_ctx_set_pix_fmt(AVCodecContext* ctx, enum AVPixelFormat fmt) { ctx->pix_fmt = fmt; }
void avcodec_ctx_set_gop_size(AVCodecContext* ctx, int gop) { ctx->gop_size = gop; }
void avcodec_ctx_set_bit_rate(AVCodecContext* ctx, int64_t br) { ctx->bit_rate = br; }
void avcodec_ctx_set_sample_rate(AVCodecContext* ctx, int sr) { ctx->sample_rate = sr; }
void avcodec_ctx_set_ch_layout(AVCodecContext* ctx, uint64_t ch_mask) { av_channel_layout_from_mask(&ctx->ch_layout, ch_mask); }
void avcodec_ctx_set_sample_fmt(AVCodecContext* ctx, int fmt) { ctx->sample_fmt = fmt; }
void avcodec_set_hw_device_ctx(AVCodecContext* ctx, AVBufferRef* hw) { ctx->hw_device_ctx = hw; }
void avcodec_set_thread_count(AVCodecContext* ctx, int c) { ctx->thread_count = c; }

AVStream* avformat_get_stream(AVFormatContext* ctx, int idx) { return ctx->streams[idx]; }
AVRational avstream_get_time_base(AVStream* st) { return st->time_base; }
void avstream_set_time_base(AVStream* st, AVRational tb) { st->time_base = tb; }
AVRational avstream_get_avg_frame_rate(AVStream* st) { return st->avg_frame_rate; }
int64_t avformat_get_duration(AVFormatContext* ctx) { return ctx->duration; }
AVCodecParameters* avstream_get_codecpar(AVStream* st) { return st->codecpar; }
AVCodecParameters* avstream_get_codecpar_mut(AVStream* st) { return st->codecpar; }

enum AVCodecID avcodecpar_get_codec_id(AVCodecParameters* par) { return par->codec_id; }
int avcodecpar_get_width(AVCodecParameters* par) { return par->width; }
int avcodecpar_get_height(AVCodecParameters* par) { return par->height; }

int av_packet_stream_index(AVPacket* pkt) { return pkt->stream_index; }
int64_t av_packet_pts(AVPacket* pkt) { return pkt->pts; }
int64_t av_packet_duration(AVPacket* pkt) { return pkt->duration; }

int av_frame_get_width(AVFrame* f) { return f->width; }
int av_frame_get_height(AVFrame* f) { return f->height; }
int64_t av_frame_get_pts(AVFrame* f) { return f->pts; }
void av_frame_set_pts(AVFrame* f, int64_t pts) { f->pts = pts; }
int* av_frame_get_linesize(AVFrame* f) { return f->linesize; }
int av_frame_get_format(AVFrame* f) { return f->format; }
int av_frame_get_nb_samples(AVFrame* f) { return f->nb_samples; }
uint8_t** av_frame_get_data(AVFrame* f) { return f->data; }
int av_frame_get_sample_rate(AVFrame* f) { return f->sample_rate; }

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

