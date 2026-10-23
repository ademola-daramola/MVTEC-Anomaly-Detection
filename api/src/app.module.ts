import { Controller, Get, Module } from '@nestjs/common';
import { BullModule } from '@nestjs/bullmq';
import { InspectionsModule } from './inspections/inspections.module';

@Controller('health')
class HealthController {
  @Get()
  health() {
    return { status: 'ok' };
  }
}

@Module({
  imports: [
    BullModule.forRoot({
      connection: {
        host: process.env.REDIS_HOST ?? 'localhost',
        port: Number(process.env.REDIS_PORT ?? 6379),
      },
    }),
    InspectionsModule,
  ],
  controllers: [HealthController],
})
export class AppModule {}
