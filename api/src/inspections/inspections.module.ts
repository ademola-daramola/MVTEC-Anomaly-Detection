import { Module } from '@nestjs/common';
import { BullModule } from '@nestjs/bullmq';
import { InspectionsController } from './inspections.controller';
import { InspectionsService, QUEUE_NAME } from './inspections.service';

@Module({
  imports: [BullModule.registerQueue({ name: QUEUE_NAME })],
  controllers: [InspectionsController],
  providers: [InspectionsService],
})
export class InspectionsModule {}
