import { BadRequestException, Injectable, NotFoundException } from '@nestjs/common';
import { InjectQueue } from '@nestjs/bullmq';
import { Queue } from 'bullmq';
import { randomUUID } from 'crypto';
import { promises as fs } from 'fs';
import { join, resolve } from 'path';

export const QUEUE_NAME = 'inspections';

export const CATEGORIES = [
  'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
  'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
];

const EXT_BY_MIME: Record<string, string> = {
  'image/png': 'png',
  'image/jpeg': 'jpg',
};

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

@Injectable()
export class InspectionsService {
  private readonly storageDir = process.env.STORAGE_DIR ?? './storage';

  constructor(@InjectQueue(QUEUE_NAME) private readonly queue: Queue) {}

  async create(file: Express.Multer.File | undefined, category: string) {
    if (!file) throw new BadRequestException("Field 'image' is required");
    if (!CATEGORIES.includes(category)) {
      throw new BadRequestException(`'category' must be one of: ${CATEGORIES.join(', ')}`);
    }
    const ext = EXT_BY_MIME[file.mimetype];
    if (!ext) throw new BadRequestException('Only PNG and JPEG images are accepted');

    const id = randomUUID();
    const uploads = join(this.storageDir, 'uploads');
    await fs.mkdir(uploads, { recursive: true });
    const imagePath = join(uploads, `${id}.${ext}`);
    await fs.writeFile(imagePath, file.buffer);

    await this.queue.add(
      'inspect',
      { image_path: imagePath, category },
      {
        jobId: id,
        attempts: 3,
        backoff: { type: 'exponential', delay: 2000 },
        removeOnComplete: { age: 24 * 3600 },
        removeOnFail: { age: 24 * 3600 },
      },
    );
    return { job_id: id, status: 'queued' };
  }

  async get(id: string) {
    if (!UUID_RE.test(id)) throw new NotFoundException('Unknown job');
    const job = await this.queue.getJob(id);
    if (!job) throw new NotFoundException('Unknown job');

    const state = await job.getState();
    const base = {
      job_id: id,
      category: job.data.category,
      created_at: new Date(job.timestamp).toISOString(),
      finished_at: job.finishedOn ? new Date(job.finishedOn).toISOString() : null,
    };

    if (state === 'completed') {
      const r = job.returnvalue;
      return {
        ...base,
        status: 'done',
        score: r.score,
        threshold: r.threshold,
        is_anomalous: r.is_anomalous,
        model_version: r.model_version,
        latency_ms: r.latency_ms,
        heatmap_url: `/inspections/${id}/heatmap`,
      };
    }
    if (state === 'failed') return { ...base, status: 'failed', error: job.failedReason };
    if (state === 'active') return { ...base, status: 'processing' };
    return { ...base, status: 'queued' };
  }

  heatmapPath(id: string): string {
    if (!UUID_RE.test(id)) throw new NotFoundException('Unknown job');
    // resolve() keeps an absolute STORAGE_DIR (e.g. /storage in Docker) and anchors a relative one to cwd
    return resolve(this.storageDir, 'heatmaps', `${id}.png`);
  }
}
