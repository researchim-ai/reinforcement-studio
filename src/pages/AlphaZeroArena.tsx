import { useCallback, useEffect, useState } from 'react'
import { toast } from 'sonner'
import { Loader2, Swords, RotateCcw } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Select } from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { useGames, useOpponents } from '@/api/hooks'
import { api } from '@/api/client'
import type { ArenaState } from '@/api/types'
import { cn } from '@/lib/utils'

export function AlphaZeroArena() {
  const { data: gamesData } = useGames()
  const games = gamesData?.games ?? []

  const [gameId, setGameId] = useState('')
  const [opponentKey, setOpponentKey] = useState('')
  const [humanFirst, setHumanFirst] = useState(true)
  const [session, setSession] = useState<ArenaState | null>(null)
  const [loading, setLoading] = useState(false)

  const { data: opponentsData } = useOpponents(gameId || undefined)
  const opponents = opponentsData?.opponents ?? []

  useEffect(() => {
    if (!gameId && games.length > 0) setGameId(games[0].id)
  }, [games, gameId])

  useEffect(() => {
    if (opponents.length > 0 && !opponents.some((o) => `${o.source}:${o.id}` === opponentKey)) {
      setOpponentKey(`${opponents[0].source}:${opponents[0].id}`)
    }
  }, [opponents, opponentKey])

  const currentGame = games.find((g) => g.id === gameId)

  const startGame = useCallback(async () => {
    if (!gameId || !opponentKey) return
    const [source, id] = opponentKey.split(':')
    setLoading(true)
    try {
      const state = await api.newArenaSession({
        game_id: gameId,
        opponent_id: id,
        opponent_source: source,
        human_first: humanFirst,
      })
      setSession(state)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Не удалось начать игру')
    } finally {
      setLoading(false)
    }
  }, [gameId, opponentKey, humanFirst])

  const makeMove = useCallback(
    async (action: number) => {
      if (!session || session.done || !session.legal_actions.includes(action)) return
      if (session.current_player !== session.human_player) return
      setLoading(true)
      try {
        const state = await api.arenaMove(session.session_id, action)
        setSession(state)
      } catch (err) {
        toast.error(err instanceof Error ? err.message : 'Ход невозможен')
      } finally {
        setLoading(false)
      }
    },
    [session],
  )

  const isDropMode = gameId === 'connect_four'

  const cellAction = (row: number, col: number, cols: number) => (isDropMode ? col : row * cols + col)

  const resultLabel = () => {
    if (!session?.done) return null
    if (session.winner === 0) return 'Ничья'
    return session.winner === session.human_player ? 'Ты победил!' : 'Победил AI'
  }

  return (
    <div className="mx-auto max-w-4xl space-y-6 p-8">
      <div className="space-y-1">
        <h2 className="text-2xl font-semibold">AlphaZero Arena</h2>
        <p className="text-sm text-muted-foreground">
          Сыграй против случайного бота или своего обученного AlphaZero-агента.
        </p>
      </div>

      <Card>
        <CardContent className="flex flex-wrap items-end gap-4 p-4">
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Игра</Label>
            <Select
              className="w-48"
              value={gameId}
              onChange={(e) => setGameId(e.target.value)}
              options={games.map((g) => ({ value: g.id, label: `${g.name} (${g.rows}×${g.cols})` }))}
            />
          </div>
          <div className="space-y-1">
            <Label className="text-xs text-muted-foreground">Соперник</Label>
            <Select
              className="w-56"
              value={opponentKey}
              onChange={(e) => setOpponentKey(e.target.value)}
              options={opponents.map((o) => ({ value: `${o.source}:${o.id}`, label: o.label }))}
            />
          </div>
          <div className="flex items-center gap-2">
            <Label className="text-xs text-muted-foreground">Хожу первым</Label>
            <Switch checked={humanFirst} onCheckedChange={setHumanFirst} />
          </div>
          <Button onClick={startGame} disabled={loading || !gameId || !opponentKey}>
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Swords className="h-4 w-4" />}
            Новая игра
          </Button>
        </CardContent>
      </Card>

      {session && currentGame && (
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="text-sm">{currentGame.name}</CardTitle>
            <div className="flex items-center gap-2">
              {session.done ? (
                <Badge variant={session.winner === session.human_player ? 'success' : session.winner === 0 ? 'secondary' : 'destructive'}>
                  {resultLabel()}
                </Badge>
              ) : (
                <Badge variant={session.current_player === session.human_player ? 'success' : 'secondary'}>
                  {session.current_player === session.human_player ? 'Твой ход' : 'AI думает...'}
                </Badge>
              )}
              <Button variant="ghost" size="icon" onClick={startGame}>
                <RotateCcw className="h-4 w-4" />
              </Button>
            </div>
          </CardHeader>
          <CardContent className="flex flex-col items-center gap-4">
            <div
              className="inline-grid gap-1"
              style={{ gridTemplateColumns: `repeat(${session.cols}, minmax(0, 1fr))` }}
            >
              {session.board.map((rowVals, row) =>
                rowVals.map((cell, col) => {
                  const action = cellAction(row, col, session.cols)
                  const isLegal = !session.done && session.legal_actions.includes(action) && session.current_player === session.human_player
                  const prob = session.ai_info?.visit_probs?.[String(action)]
                  return (
                    <button
                      key={`${row}-${col}`}
                      onClick={() => makeMove(action)}
                      disabled={!isLegal}
                      className={cn(
                        'relative flex h-12 w-12 items-center justify-center rounded-md border border-border text-lg font-bold transition-colors',
                        isLegal ? 'cursor-pointer hover:border-primary hover:bg-primary/10' : 'cursor-default',
                        cell === 0 ? 'bg-muted/40' : 'bg-muted',
                      )}
                    >
                      {cell === 1 && <span className="text-primary">●</span>}
                      {cell === -1 && <span className="text-warning">○</span>}
                      {prob != null && cell === 0 && !session.done && (
                        <span className="absolute bottom-0.5 right-0.5 text-[8px] text-muted-foreground">
                          {(prob * 100).toFixed(0)}
                        </span>
                      )}
                    </button>
                  )
                }),
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              ● — {session.human_player === 1 ? 'ты' : 'AI'} · ○ — {session.human_player === -1 ? 'ты' : 'AI'}
              {isDropMode ? ' · клик по любой клетке столбца бросает фишку' : ''}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
