import {
  Body, Controller, Get, HttpCode, NotFoundException, Param, Post, Res,
  UploadedFile, UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { Response } from 'express';
import { existsSync } from 'fs';
import { InspectionsService } from './inspections.service';

@Controller('inspections')
export class InspectionsController {
  constructor(private readonly service: InspectionsService) {}

  @Post()
  @HttpCode(202)
  @UseInterceptors(FileInterceptor('image', { limits: { fileSize: 10 * 1024 * 1024 } }))
  create(@UploadedFile() file: Express.Multer.File, @Body('category') category: string) {
    return this.service.create(file, category);
  }

  @Get(':id')
  get(@Param('id') id: string) {
    return this.service.get(id);
  }

  @Get(':id/heatmap')
  heatmap(@Param('id') id: string, @Res() res: Response) {
    const path = this.service.heatmapPath(id);
    if (!existsSync(path)) throw new NotFoundException('Heatmap not ready');
    // The path is built server-side from a validated UUID, so it is safe to serve.
    return res.sendFile(path);
  }
}
