import { extendTheme, type ThemeConfig } from '@chakra-ui/react'
import { mode } from '@chakra-ui/theme-tools'
import { defineStyle, defineStyleConfig } from '@chakra-ui/react'

const brandPrimary = defineStyle((props) => {
  const color = mode(`brand.700`, `whiteAlpha.900`)(props)
  return {
    color: color,
    border: '1px solid',
    borderColor: color,
    bg: 'transparent',
    _hover: {
      bg: color,
      color: mode(`white`, `gray.900`)(props),
      borderColor: color,
    },
    _active: {
      bg: color,
      color: mode(`white`, `gray.900`)(props),
      borderColor: color,
    },
    _focus: {
      boxShadow: 'none',
    },
  }
})

export const buttonTheme = defineStyleConfig({
  variants: { brandPrimary },
  defaultProps: { variant: 'brandPrimary' },
})

const fonts = { heading: `'IBM Plex Sans', sans-serif`,
                body: `-apple-system, system-ui, BlinkMacSystemFont, 'Segoe UI', Roboto, Ubuntu, sans-serif`,
                mono: `'Menlo', monospace` }

const config: ThemeConfig = {
  initialColorMode: 'system',
  useSystemColorMode: true
}

const theme = extendTheme({
  config,
  fonts,
  colors: {
    black: '#16161D',
    brand: {
      /*
      500: '#6baeff',
      600: '#074d9d',
      700: '#1f5b93',
      */
     500: '#516d8a',
     600: '#516d8a',
     700: '#516d8a',
    }
  },
  components: {
    Button: buttonTheme,
  },
})

export default theme
