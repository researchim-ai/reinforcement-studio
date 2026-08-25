import Docker from 'dockerode'
import fs from 'fs'
import path from 'path'

const IMAGE_NAME_CPU = 'rl-studio:latest'
const IMAGE_NAME_GPU = 'rl-studio:gpu'
const CONTAINER_NAME = 'rl-studio-backend'
const CONTAINER_PORT = 8000

export type BuildProgress = (line: string) => void

export interface DockerStatus {
  running: boolean
  containerId?: string
  hostPort?: number
  error?: string
}

export class DockerManager {
  private docker: Docker
  private containerId: string | null = null
  private hostPort: number | null = null
  private projectRoot: string
  private buildStream: NodeJS.ReadableStream | null = null

  constructor(projectRoot?: string) {
    this.docker = new Docker()
    this.projectRoot = projectRoot ?? path.resolve(__dirname, '..', '..')
  }

  // -----------------------------------------------------------------------
  // Availability
  // -----------------------------------------------------------------------

  async isAvailable(): Promise<boolean> {
    try {
      await this.docker.ping()
      return true
    } catch {
      return false
    }
  }

  hasDockerfile(gpu = false): boolean {
    return fs.existsSync(path.join(this.projectRoot, gpu ? 'Dockerfile.gpu' : 'Dockerfile'))
  }

  async imageExists(gpu = false): Promise<boolean> {
    try {
      const images = await this.docker.listImages({
        filters: { reference: [gpu ? IMAGE_NAME_GPU : IMAGE_NAME_CPU] },
      })
      return images.length > 0
    } catch {
      return false
    }
  }

  // -----------------------------------------------------------------------
  // Status
  // -----------------------------------------------------------------------

  async getStatus(): Promise<DockerStatus> {
    try {
      const containers = await this.docker.listContainers({
        all: true,
        filters: { name: [CONTAINER_NAME] },
      })

      if (containers.length > 0) {
        const c = containers[0]
        this.containerId = c.Id
        const mapped = c.Ports?.find((p) => p.PrivatePort === CONTAINER_PORT)
        const port = mapped?.PublicPort ?? this.hostPort ?? undefined
        if (port) this.hostPort = port
        return {
          running: c.State === 'running',
          containerId: c.Id,
          hostPort: port,
        }
      }
      return { running: false }
    } catch (error) {
      return { running: false, error: String(error) }
    }
  }

  // -----------------------------------------------------------------------
  // Build
  // -----------------------------------------------------------------------

  async buildImage(onProgress?: BuildProgress, gpu = false, torchCudaChannel?: string): Promise<{ success: boolean; error?: string }> {
    const dockerfile = gpu ? 'Dockerfile.gpu' : 'Dockerfile'
    if (!this.hasDockerfile(gpu)) {
      return { success: false, error: `${dockerfile} not found at ${this.projectRoot}/${dockerfile}` }
    }

    try {
      const stream = await this.docker.buildImage(
        { context: this.projectRoot, src: ['.'] },
        {
          t: gpu ? IMAGE_NAME_GPU : IMAGE_NAME_CPU,
          dockerfile,
          // Only meaningful for Dockerfile.gpu (plain Dockerfile has no
          // matching ARG) — which CUDA wheel channel torch gets installed
          // from, picked from the host's driver right before the build
          // (see electron/gpu.ts's pickTorchCudaChannel).
          buildargs: gpu && torchCudaChannel ? { TORCH_CUDA_CHANNEL: torchCudaChannel } : undefined,
        },
      )
      this.buildStream = stream

      await new Promise<void>((resolve, reject) => {
        this.docker.modem.followProgress(
          stream,
          (err: Error | null) => (err ? reject(err) : resolve()),
          (event: Record<string, unknown>) => {
            if (!onProgress) return
            const streamLine = event.stream as string | undefined
            const status = event.status as string | undefined
            const progress = event.progress as string | undefined
            const errorDetail = event.errorDetail as { message?: string } | undefined
            const text =
              errorDetail?.message ||
              (streamLine ? streamLine.trimEnd() : '') ||
              (status ? `${status}${progress ? ' ' + progress : ''}` : '')
            if (text) onProgress(text)
          },
        )
      })

      return { success: true }
    } catch (error) {
      return { success: false, error: String(error) }
    } finally {
      this.buildStream = null
    }
  }

  /** Aborts an in-progress buildImage() call (destroys its HTTP connection
   * to the daemon, which the daemon treats as a client disconnect and stops
   * the build) — lets the user bail out of e.g. an Auto-mode boot that
   * picked Docker and is now downloading a multi-GB CUDA image, instead of
   * being stuck watching it with no escape but force-quitting the app. */
  cancelBuild(): void {
    try {
      (this.buildStream as { destroy?: () => void } | null)?.destroy?.()
    } catch { /* ignore */ }
    this.buildStream = null
  }

  // -----------------------------------------------------------------------
  // Run
  // -----------------------------------------------------------------------

  async ensureContainerRemoved(): Promise<void> {
    try {
      const containers = await this.docker.listContainers({
        all: true,
        filters: { name: [CONTAINER_NAME] },
      })
      for (const c of containers) {
        const container = this.docker.getContainer(c.Id)
        try {
          if (c.State === 'running') await container.stop({ t: 5 })
        } catch { /* ignore */ }
        try {
          await container.remove({ force: true })
        } catch { /* ignore */ }
      }
    } catch { /* ignore */ }
  }

  async startContainer(opts: { hostPort: number; gpu?: boolean }): Promise<{
    success: boolean
    hostPort?: number
    error?: string
  }> {
    try {
      await this.ensureContainerRemoved()
      for (const dir of ['rl_core/.runs', 'rl_core/checkpoints']) {
        fs.mkdirSync(path.join(this.projectRoot, dir), { recursive: true })
      }

      const deviceRequests = opts.gpu ? [{ Count: -1, Capabilities: [['gpu']] }] : undefined

      const container = await this.docker.createContainer({
        Image: opts.gpu ? IMAGE_NAME_GPU : IMAGE_NAME_CPU,
        name: CONTAINER_NAME,
        Cmd: ['python', '-m', 'uvicorn', 'backend.api:app', '--host', '0.0.0.0', '--port', String(CONTAINER_PORT)],
        ExposedPorts: { [`${CONTAINER_PORT}/tcp`]: {} },
        HostConfig: {
          PortBindings: {
            [`${CONTAINER_PORT}/tcp`]: [{ HostPort: String(opts.hostPort) }],
          },
          Binds: [
            `${this.projectRoot}/rl_core/.runs:/app/rl_core/.runs`,
            `${this.projectRoot}/rl_core/checkpoints:/app/rl_core/checkpoints`,
          ],
          ShmSize: 2 * 1024 * 1024 * 1024,
          DeviceRequests: deviceRequests,
          RestartPolicy: { Name: 'unless-stopped' },
        },
        Env: [
          'PYTHONUNBUFFERED=1',
          'RL_STUDIO_ROOT=/app/rl_core',
          ...(opts.gpu ? ['NVIDIA_VISIBLE_DEVICES=all', 'NVIDIA_DRIVER_CAPABILITIES=all'] : []),
        ],
      })

      await container.start()
      this.containerId = container.id
      this.hostPort = opts.hostPort
      return { success: true, hostPort: opts.hostPort }
    } catch (error) {
      return { success: false, error: String(error) }
    }
  }

  async stopContainer(): Promise<{ success: boolean; error?: string }> {
    try {
      const status = await this.getStatus()
      if (!status.running || !status.containerId) return { success: true }

      const container = this.docker.getContainer(status.containerId)
      await container.stop({ t: 10 })
      return { success: true }
    } catch (error) {
      return { success: false, error: String(error) }
    }
  }

  async waitForHealth(
    hostPort: number,
    opts: { timeoutMs?: number; onTick?: (attempt: number) => void } = {},
  ): Promise<boolean> {
    const timeoutMs = opts.timeoutMs ?? 120_000
    const start = Date.now()
    let attempt = 0
    while (Date.now() - start < timeoutMs) {
      attempt += 1
      opts.onTick?.(attempt)
      try {
        const res = await fetch(`http://127.0.0.1:${hostPort}/api/system/health`)
        if (res.ok) return true
      } catch { /* not ready */ }
      await new Promise((r) => setTimeout(r, 1000))
    }
    return false
  }

  async getContainerLogs(tail = 200): Promise<string> {
    try {
      const status = await this.getStatus()
      if (!status.containerId) return ''

      const container = this.docker.getContainer(status.containerId)
      const logs = await container.logs({ stdout: true, stderr: true, tail, timestamps: true })
      return logs.toString()
    } catch {
      return ''
    }
  }
}
