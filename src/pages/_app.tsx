import { Chakra } from "../Chakra";
import { AppProps } from 'next/app'

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <Chakra cookies={pageProps.cookies}>
      <Component {...pageProps} />
    </Chakra>
  )
}

export default MyApp

export { getServerSideProps } from "../Chakra";